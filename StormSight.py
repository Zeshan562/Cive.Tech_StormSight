"""
StormSight - Stormwater Runoff Calculator & Visualizer (Condensed)
HackED 2026 | Rational Method · Storm Analysis · Spatial Heatmap · GeoTIFF Drainage
"""
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.colors as mcolors
import numpy as np, os, warnings
from scipy.ndimage import gaussian_filter, label, maximum_filter, center_of_mass, zoom

# ── Constants ────────────────────────────────────────────────
RUNOFF_C = {"Downtown/Commercial": 0.90, "Urban Residential": 0.70, "Suburban Residential": 0.45,
            "Industrial/Warehouse": 0.65, "Parks/Lawns": 0.25, "Agricultural": 0.35,
            "Forest/Woodland": 0.15, "Wetland/Marshy": 0.10}
SOIL_G = {"A – Sandy": 1.0, "B – Silt": 0.85, "C – Clay Loam": 0.65, "D – Clay": 0.50}
IDF = {"2yr": {5:85,10:65,15:52,30:35,60:22,120:14}, "5yr": {5:105,10:80,15:64,30:43,60:28,120:18},
       "10yr": {5:120,10:92,15:74,30:50,60:33,120:21}, "25yr": {5:142,10:109,15:88,30:60,60:40,120:26},
       "100yr": {5:175,10:134,15:108,30:74,60:50,120:33}}
RISK_T = {"Low":(0,0.5), "Moderate":(0.5,2), "High":(2,6), "Critical":(6,float("inf"))}
RISK_CLR = {"Low":"#2ecc71","Moderate":"#f39c12","High":"#e74c3c","Critical":"#8e44ad"}
BG,PNL,ACC,ACC2,TXT,MUT,BRD,WRN,GLD = "#0d1117","#161b22","#58a6ff","#3fb950","#e6edf3","#8b949e","#30363d","#f85149","#e3b341"

# ── Engineering Calculations ─────────────────────────────────
def kirpich_tc(L, S_pct, sf):
    S = max(S_pct, 0.1) / 100; return max(0.0195 * L**0.77 * S**-0.385 * sf, 5.0)
def idf_intensity(tc, rp):
    d = sorted(IDF[rp]); v = [IDF[rp][k] for k in d]
    if tc <= d[0]: return v[0]
    if tc >= d[-1]: return v[-1]
    for i in range(len(d)-1):
        if d[i] <= tc <= d[i+1]: return v[i] + (tc-d[i])/(d[i+1]-d[i])*(v[i+1]-v[i])
    return v[-1]
def rational_Q(C, I, A): return C * (I/3.6e6) * A * 1e4
def classify(Q):
    for l,(lo,hi) in RISK_T.items():
        if lo <= Q < hi: return l
    return "Critical"

# ── D8 Flow Engine ───────────────────────────────────────────
def breach(dem, iters=12):
    f = dem.astype(float).copy()
    for _ in range(iters):
        p = np.pad(f,1,mode="edge")
        nm = np.minimum.reduce([p[:-2,:-2],p[:-2,1:-1],p[:-2,2:],p[1:-1,:-2],p[1:-1,2:],p[2:,:-2],p[2:,1:-1],p[2:,2:]])
        s = f < nm; f[s] = nm[s]+0.001
    return f
def d8_flow(dem):
    DX,DY = [1,1,0,-1,-1,-1,0,1],[0,-1,-1,-1,0,1,1,1]; DS = [1,1.414,1,1.414,1,1.414,1,1.414]
    ny,nx = dem.shape; bd = np.full((ny,nx),-np.inf); fx = np.zeros((ny,nx),np.int8); fy = fx.copy()
    pad = np.pad(dem,1,mode="edge")
    for dx,dy,ds in zip(DX,DY,DS):
        r0,r1 = max(0,dy),ny+min(0,dy); c0,c1 = max(0,dx),nx+min(0,dx)
        nbr = np.full((ny,nx),np.nan); nbr[r0:r1,c0:c1] = pad[1+max(0,-dy):1+max(0,-dy)+(r1-r0),1+max(0,-dx):1+max(0,-dx)+(c1-c0)]
        drop = (dem-nbr)/ds; b = drop>bd; bd[b]=drop[b]; fx[b]=dx; fy[b]=dy
    return fx, fy
def flow_acc(fx, fy, passes=8):
    ny,nx = fx.shape; acc = np.ones((ny,nx),float)
    iy,ix = np.meshgrid(np.arange(ny),np.arange(nx),indexing="ij")
    ni,nj = np.clip(iy+fy.astype(int),0,ny-1), np.clip(ix+fx.astype(int),0,nx-1); mv = (ni!=iy)|(nj!=ix)
    for _ in range(passes): n = acc.copy(); np.add.at(n,(ni[mv],nj[mv]),acc[mv]); acc = n
    return acc

def read_tif(path):
    """Load GeoTIFF, downsample, clean NaNs. Returns (elev, transform, crs)."""
    import rasterio
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with rasterio.open(path) as src:
            e = src.read(1).astype(np.float32); nd = src.nodata; t,crs = src.transform, src.crs
    if nd is not None: e = np.where(e == nd, np.nan, e)
    if max(e.shape) > 300: e = zoom(e, 300/max(e.shape), order=1)
    m = np.isnan(e)
    if m.any(): e = np.where(m, gaussian_filter(np.nan_to_num(e,nan=0),3), e)
    return e, t, crs

# ── GUI ──────────────────────────────────────────────────────
class App:
    def __init__(self, root):
        self.root = root; self.tif_path = None
        root.title("StormSight – HackED 2026"); root.configure(bg=BG); root.geometry("1280x800")
        s = ttk.Style(); s.theme_use("clam")
        for n,b,f in [("TFrame",BG,TXT),("TLabel",BG,TXT),("TNotebook",BG,MUT)]:
            s.configure(n, background=b, foreground=f, font=("Consolas",10))
        s.configure("TNotebook.Tab", background=PNL, foreground=MUT, font=("Consolas",10,"bold"), padding=[12,5])
        s.map("TNotebook.Tab", background=[("selected",BG)], foreground=[("selected",ACC)])
        s.configure("TCombobox", fieldbackground=BRD, foreground=TXT, font=("Consolas",10))
        s.map("TCombobox", fieldbackground=[("readonly",BRD)])
        for nm,c in [("A.TButton",ACC),("S.TButton",GLD),("H.TButton","#da3633"),("T.TButton","#6e40c9")]:
            s.configure(nm, background=c, foreground="#0d1117" if c not in ("#da3633","#6e40c9") else "#fff",
                        font=("Consolas",11,"bold"), padding=8)
        self._build()

    def _mk(self, p, lbl, val, u=""):
        r = tk.Frame(p,bg=PNL); r.pack(fill="x",pady=2)
        tk.Label(r,text=lbl,font=("Consolas",9),fg=MUT,bg=PNL).pack(anchor="w")
        f = tk.Frame(r,bg=PNL); f.pack(fill="x"); v = tk.StringVar(value=str(val))
        tk.Entry(f,textvariable=v,font=("Consolas",10),bg=BRD,fg=TXT,insertbackground=TXT,relief="flat",bd=3,width=14).pack(side="left")
        if u: tk.Label(f,text=f" {u}",font=("Consolas",9),fg=MUT,bg=PNL).pack(side="left")
        return v
    def _cb(self, p, lbl, opts, i=0):
        r = tk.Frame(p,bg=PNL); r.pack(fill="x",pady=2)
        tk.Label(r,text=lbl,font=("Consolas",9),fg=MUT,bg=PNL).pack(anchor="w")
        v = tk.StringVar(value=opts[i]); ttk.Combobox(r,textvariable=v,values=opts,state="readonly",font=("Consolas",9)).pack(fill="x")
        return v

    def _build(self):
        hdr = tk.Frame(self.root,bg=BG,pady=6); hdr.pack(fill="x",padx=16)
        tk.Label(hdr,text="⛈ StormSight",font=("Consolas",18,"bold"),fg=ACC,bg=BG).pack(side="left")
        tk.Label(hdr,text="  Stormwater Runoff Calculator | HackED 2026",font=("Consolas",10),fg=MUT,bg=BG).pack(side="left")
        body = tk.Frame(self.root,bg=BG); body.pack(fill="both",expand=True,padx=16,pady=(0,8))
        left = tk.Frame(body,bg=BG,width=355); left.pack(side="left",fill="y",padx=(0,10)); left.pack_propagate(False)
        right = tk.Frame(body,bg=BG); right.pack(side="left",fill="both",expand=True)
        # Watershed inputs
        bx = tk.Frame(left,bg=PNL,highlightbackground=BRD,highlightthickness=1); bx.pack(fill="x",pady=(0,4))
        tk.Label(bx,text="[ WATERSHED ]",font=("Consolas",10,"bold"),fg=ACC,bg=PNL,pady=6).pack()
        inn = tk.Frame(bx,bg=PNL,padx=10); inn.pack(fill="x")
        self.v_a = self._mk(inn,"Drainage Area","10.0","ha"); self.v_l = self._mk(inn,"Flow Length","500","m")
        self.v_s = self._mk(inn,"Slope","2.0","%"); self.v_lu = self._cb(inn,"Land Use",list(RUNOFF_C),2)
        self.v_so = self._cb(inn,"Soil Group",list(SOIL_G),1); self.v_rp = self._cb(inn,"Return Period",list(IDF),2)
        tk.Frame(bx,bg=PNL,height=6).pack()
        ttk.Button(left,text="▶  CALCULATE RUNOFF",style="A.TButton",command=self.calc).pack(fill="x",pady=(0,3))
        # Storm inputs
        sb = tk.Frame(left,bg=PNL,highlightbackground=BRD,highlightthickness=1); sb.pack(fill="x",pady=(0,3))
        tk.Label(sb,text="[ STORM ANALYSIS ]",font=("Consolas",10,"bold"),fg=GLD,bg=PNL,pady=4).pack()
        si = tk.Frame(sb,bg=PNL,padx=10); si.pack(fill="x")
        tk.Label(si,text="Cumulative rainfall (one/line):",font=("Consolas",8),fg=MUT,bg=PNL).pack(anchor="w")
        self.stxt = tk.Text(si,height=3,font=("Consolas",9),bg=BRD,fg=TXT,relief="flat",bd=3)
        self.stxt.pack(fill="x",pady=2); self.stxt.insert("end","0\n3.1\n10.5\n28.0\n45.2\n60.0\n70.0\n74.0\n75.5")
        dr = tk.Frame(si,bg=PNL); dr.pack(fill="x",pady=1)
        tk.Label(dr,text="Δt:",font=("Consolas",9),fg=MUT,bg=PNL).pack(side="left")
        self.v_dt = tk.StringVar(value="0.25")
        tk.Entry(dr,textvariable=self.v_dt,font=("Consolas",9),bg=BRD,fg=TXT,relief="flat",bd=3,width=6).pack(side="left",padx=3)
        self.v_dtu = tk.StringVar(value="hr"); ttk.Combobox(dr,textvariable=self.v_dtu,values=["hr","min"],state="readonly",width=4).pack(side="left",padx=2)
        self.v_ru = tk.StringVar(value="mm"); ur = tk.Frame(si,bg=PNL); ur.pack(fill="x")
        tk.Label(ur,text="Units:",font=("Consolas",9),fg=MUT,bg=PNL).pack(side="left")
        for u in ["mm","in"]: tk.Radiobutton(ur,text=u,variable=self.v_ru,value=u,bg=PNL,fg=TXT,selectcolor=BRD,font=("Consolas",9)).pack(side="left",padx=2)
        tk.Frame(sb,bg=PNL,height=3).pack()
        ttk.Button(left,text="📊  MASS CURVE & HYETOGRAPH",style="S.TButton",command=self.storm).pack(fill="x",pady=(0,3))
        ttk.Button(left,text="🌡  SPATIAL HEATMAP",style="H.TButton",command=self.heatmap).pack(fill="x",pady=(0,3))
        ttk.Button(left,text="🗺  LOAD GeoTIFF + DRAINAGE",style="T.TButton",command=self.load_tif).pack(fill="x",pady=(0,3))
        # Results + Tabs
        rf = tk.Frame(right,bg=PNL,highlightbackground=BRD,highlightthickness=1); rf.pack(fill="x",pady=(0,4))
        self.lQ = tk.Label(rf,text="Q = —",font=("Consolas",16,"bold"),fg=ACC2,bg=PNL); self.lQ.pack(side="left",padx=14)
        self.lT = tk.Label(rf,text="Tc = —",font=("Consolas",11),fg=TXT,bg=PNL); self.lT.pack(side="left",padx=10)
        self.lV = tk.Label(rf,text="Vol = —",font=("Consolas",11),fg=TXT,bg=PNL); self.lV.pack(side="left",padx=10)
        self.lR = tk.Label(rf,text="Risk: —",font=("Consolas",12,"bold"),fg=WRN,bg=PNL); self.lR.pack(side="right",padx=14)
        self.lC = tk.Label(rf,text="",font=("Consolas",9),fg=MUT,bg=PNL); self.lC.pack(side="right",padx=10)
        self.nb = ttk.Notebook(right); self.nb.pack(fill="both",expand=True)
        self.figs, self.canvs = [], []
        for name in ["Runoff Analysis","Storm Analysis","Spatial Heatmap","Drainage Planning (GeoTIFF)"]:
            tab = tk.Frame(self.nb,bg=BG); self.nb.add(tab,text=f"  {name}  ")
            fig = Figure(figsize=(14,8) if "GeoTIFF" in name else (9,5), facecolor=BG)
            c = FigureCanvasTkAgg(fig,master=tab); c.get_tk_widget().pack(fill="both",expand=True)
            self.figs.append(fig); self.canvs.append(c)

    def _sax(self, ax, title="", gold=False):
        ax.set_facecolor(PNL); [sp.set_edgecolor(BRD) for sp in ax.spines.values()]
        ax.tick_params(colors=MUT,labelsize=7); ax.xaxis.label.set_color(MUT); ax.yaxis.label.set_color(MUT)
        if title: ax.set_title(title,color=GLD if gold else ACC,fontsize=9)
        ax.grid(True,color=BRD,lw=0.4,alpha=0.5)

    # ── Tab 1: Runoff ────────────────────────────────────────
    def calc(self):
        try: A,L,S = float(self.v_a.get()),float(self.v_l.get()),float(self.v_s.get())
        except ValueError: messagebox.showerror("Error","Enter valid numbers."); return
        C = RUNOFF_C[self.v_lu.get()]; sf = SOIL_G[self.v_so.get()]; rp = self.v_rp.get()
        tc = kirpich_tc(L,S,sf); I = idf_intensity(tc,rp); Q = rational_Q(C,I,A); risk = classify(Q)
        self.lQ.config(text=f"Q = {Q:.4f} m³/s"); self.lT.config(text=f"Tc = {tc:.1f} min")
        self.lV.config(text=f"Vol ≈ {Q*tc*60:.0f} m³"); self.lC.config(text=f"C={C:.2f}  I={I:.1f} mm/hr")
        self.lR.config(text=f"Risk: {risk}",fg=RISK_CLR[risk])
        fig = self.figs[0]; fig.clear(); fig.patch.set_facecolor(BG)
        gs = fig.add_gridspec(2,3,hspace=0.45,wspace=0.35,left=0.07,right=0.97,top=0.93,bottom=0.1)
        ax1,ax2,ax3,ax4 = fig.add_subplot(gs[0,0]),fig.add_subplot(gs[0,1]),fig.add_subplot(gs[0,2]),fig.add_subplot(gs[1,:])
        for a in [ax1,ax2,ax3,ax4]: self._sax(a)
        dur = [5,10,15,30,60,120]; clrs = ["#58a6ff","#3fb950","#f0883e","#ff7b72","#d2a8ff"]
        for i,(p,d) in enumerate(IDF.items()):
            ax1.plot(dur,[d[k] for k in dur],color=clrs[i],lw=2.5 if p==rp else 0.8,alpha=1 if p==rp else 0.35,label=p,marker="o",ms=2)
        ax1.axvline(tc,color=WRN,ls="--",lw=1.2); ax1.set_xlabel("Duration (min)"); ax1.set_ylabel("I (mm/hr)")
        ax1.set_title("IDF Curves",color=ACC); ax1.legend(fontsize=6,labelcolor=MUT,facecolor=BG,edgecolor=BRD)
        areas = np.linspace(0.5,A*2.5,50); ax2.plot(areas,[rational_Q(C,I,a) for a in areas],color=ACC,lw=2)
        ax2.scatter([A],[Q],color=WRN,s=50,zorder=5); ax2.set_xlabel("Area (ha)"); ax2.set_ylabel("Q (m³/s)")
        ax2.set_title(f"Q vs Area (C={C:.2f})",color=ACC)
        rl = list(RISK_CLR); ri = rl.index(risk)
        w,_ = ax3.pie([1]*4,labels=rl,colors=[RISK_CLR[r] for r in rl],explode=[0.05 if i==ri else 0 for i in range(4)],
                       startangle=90,textprops={"color":MUT,"fontsize":7},wedgeprops={"linewidth":1,"edgecolor":BG})
        for i,ww in enumerate(w): ww.set_alpha(1.0 if i==ri else 0.25)
        ax3.set_title(f"Risk: {risk}",color=RISK_CLR[risk]); ax3.grid(False)
        tp,tb = tc,2.67*tc; t = np.linspace(0,tb*1.4,300)
        h = np.where(t<=tp,Q*t/tp,np.where(t<=tb,Q*(tb-t)/(tb-tp),0))+0.02*Q
        ax4.fill_between(t,h,0.02*Q,alpha=0.25,color=RISK_CLR[risk]); ax4.plot(t,h,color=RISK_CLR[risk],lw=2,label="Runoff")
        ax4.axvline(tp,color=WRN,ls="--",lw=1.2,label=f"Peak@{tp:.0f}min"); ax4.set_xlabel("Time (min)"); ax4.set_ylabel("Q (m³/s)")
        ax4.set_title(f"Hydrograph — {self.v_lu.get()} | {rp} | A={A}ha | Qp={Q:.4f}",color=ACC,fontsize=8)
        ax4.legend(fontsize=7,labelcolor=MUT,facecolor=BG,edgecolor=BRD,loc="upper right"); ax4.set_xlim(0,tb*1.4)
        self.canvs[0].draw(); self.nb.select(0)

    # ── Tab 2: Storm ─────────────────────────────────────────
    def _parse_storm(self):
        raw = self.stxt.get("1.0","end").strip().splitlines()
        cp = [float(v) for v in raw if v.strip()]; dt = float(self.v_dt.get())
        dt_hr = dt if self.v_dtu.get()=="hr" else dt/60; ru = self.v_ru.get()
        return cp, dt, dt_hr, ru

    def storm(self):
        try: cp,dt,dt_hr,ru = self._parse_storm()
        except ValueError: messagebox.showerror("Error","Bad data."); return
        if len(cp)<2: messagebox.showerror("Error","Need ≥2 values."); return
        iu = f"{ru}/hr"; n = len(cp); times = [i*dt for i in range(n)]
        dp = [cp[i]-cp[i-1] for i in range(1,n)]; inten = [d/dt_hr for d in dp]
        tmid = [(times[i]+times[i+1])/2 for i in range(n-1)]
        pk = max(inten); pkt = tmid[inten.index(pk)]; tot = cp[-1]-cp[0]
        self.lQ.config(text=f"P={tot:.2f}{ru}"); self.lT.config(text=f"Δt={dt}{self.v_dtu.get()}")
        self.lV.config(text=f"{n-1} intervals"); self.lC.config(text=f"Peak={pk:.3f}{iu} @t={pkt:.2f}")
        fig = self.figs[1]; fig.clear(); fig.patch.set_facecolor(BG)
        gs = fig.add_gridspec(2,1,hspace=0.5,left=0.09,right=0.97,top=0.93,bottom=0.1)
        a1,a2 = fig.add_subplot(gs[0]),fig.add_subplot(gs[1])
        for a in [a1,a2]: self._sax(a,gold=True)
        a1.plot(times,cp,color=ACC,lw=2.5,marker="o",ms=3,markerfacecolor=ACC2); a1.fill_between(times,cp,alpha=0.1,color=ACC)
        a1.set_xlabel(f"Time ({self.v_dtu.get()})"); a1.set_ylabel(f"P ({ru})"); a1.set_title("Cumulative Mass Curve",color=GLD)
        bc = [WRN if i==inten.index(pk) else ACC for i in range(len(inten))]
        a2.bar(tmid,inten,width=dt*0.85,color=bc,alpha=0.8,edgecolor=BG,lw=0.5)
        a2.axhline(pk,color=WRN,ls="--",lw=1,alpha=0.6,label=f"Peak={pk:.3f}{iu}")
        a2.set_xlabel(f"Time ({self.v_dtu.get()})"); a2.set_ylabel(f"Intensity ({iu})")
        a2.set_title(f"Hyetograph [Δt={dt}{self.v_dtu.get()}, {n-1} intervals]",color=GLD)
        a2.legend(fontsize=7,labelcolor=MUT,facecolor=BG,edgecolor=BRD)
        self.canvs[1].draw(); self.nb.select(1)

    # ── Tab 3: Heatmap ───────────────────────────────────────
    def _get_elev(self):
        if self.tif_path:
            try: e,_,_ = read_tif(self.tif_path); return e,*e.shape,os.path.basename(self.tif_path),True
            except Exception: pass
        NY=NX=120; np.random.seed(42); x,y = np.linspace(0,1,NX),np.linspace(0,1,NY); xx,yy = np.meshgrid(x,y)
        e = gaussian_filter(55-35*yy-8*xx-10*np.exp(-((xx-0.28)**2)/0.025),2.5)+gaussian_filter(np.random.randn(NY,NX)*1.2,1.2)
        return e,NY,NX,"Synthetic",False

    def heatmap(self):
        try: cp,_,dt_hr,ru = self._parse_storm(); dp=[cp[i]-cp[i-1] for i in range(1,len(cp))]; pk=max(d/dt_hr for d in dp); tot=cp[-1]-cp[0]
        except Exception: pk,tot,ru = 3.12,6.38,"in"
        elev,NY,NX,lbl,real = self._get_elev()
        en = (elev-elev.min())/max(elev.max()-elev.min(),1); gy,gx = np.gradient(elev)
        ww = np.clip(gx+gy,0,None); ww /= max(ww.max(),1e-6)
        oro = gaussian_filter(0.55+0.55*en+0.3*ww,max(NY,NX)//25)
        igrid = np.clip(oro/oro.max()*pk+gaussian_filter(np.random.randn(NY,NX)*0.04*pk,3),0,pk)
        filled = breach(elev); fx,fy = d8_flow(filled); acc = flow_acc(fx,fy,6)
        fp,mc = acc>np.percentile(acc,90), acc>np.percentile(acc,97)
        oy,ox = np.unravel_index(np.argmax(acc),acc.shape)
        fig = self.figs[2]; fig.clear(); fig.patch.set_facecolor(BG)
        gs = fig.add_gridspec(1,2,width_ratios=[1,0.05],left=0.06,right=0.96,top=0.90,bottom=0.07,wspace=0.03)
        ax,cax = fig.add_subplot(gs[0]),fig.add_subplot(gs[1])
        ax.set_facecolor(PNL); [sp.set_edgecolor(BRD) for sp in ax.spines.values()]
        im = ax.imshow(igrid,origin="lower",cmap="turbo",aspect="auto",interpolation="bilinear",vmin=0,vmax=pk)
        hs = np.gradient(elev)[1]; hs = (hs-hs.min())/max(hs.max()-hs.min(),1e-6)
        ax.imshow(hs,origin="lower",cmap="Greys_r",aspect="auto",alpha=0.18,interpolation="bilinear")
        ax.contour(fp.astype(float),levels=[0.5],colors=["#00bfff"],linewidths=0.8,alpha=0.75)
        ax.contour(mc.astype(float),levels=[0.5],colors=["white"],linewidths=1.5,alpha=0.85)
        ax.scatter([ox],[oy],color="white",s=70,zorder=7,edgecolors=BG,lw=1,marker="v")
        cs = ax.contour(igrid,levels=np.linspace(pk*0.2,pk*0.9,6),colors="white",linewidths=0.5,alpha=0.45)
        ax.clabel(cs,fmt=f"%.1f{ru}/hr",fontsize=5,colors="white")
        ax.set_title(f"Spatial Heatmap | {lbl} | Peak={pk:.2f}{ru}/hr | Total={tot:.2f}{ru}",color=GLD,fontsize=8.5)
        ax.tick_params(colors=MUT,labelsize=5)
        ax.legend(handles=[Line2D([0],[0],color="#00bfff",lw=1,label="Flow"),Line2D([0],[0],color="white",lw=1.5,label="Channels")],
                  fontsize=6,labelcolor="white",facecolor=PNL,edgecolor=BRD,loc="upper right")
        cb = fig.colorbar(im,cax=cax); cb.set_label(f"{ru}/hr",color=MUT,fontsize=7); cb.ax.tick_params(colors=MUT,labelsize=5); cb.outline.set_edgecolor(BRD)
        self.canvs[2].draw(); self.nb.select(2)

    # ── Tab 4: GeoTIFF Drainage ──────────────────────────────
    def load_tif(self):
        path = filedialog.askopenfilename(title="Select Elevation GeoTIFF",filetypes=[("GeoTIFF","*.tif *.tiff"),("All","*.*")])
        if not path: return
        self.tif_path = path
        try: elev,tf,crs = read_tif(path)
        except Exception as e: messagebox.showerror("Error",str(e)); return
        ny,nx = elev.shape
        try: C = RUNOFF_C[self.v_lu.get()]
        except: C = 0.65
        try: I_mm = IDF[self.v_rp.get()][60]
        except: I_mm = 33.0
        filled = breach(elev); fx,fy = d8_flow(filled); acc = flow_acc(fx,fy)
        pond_d = np.maximum(filled-elev.astype(float),0); pond_lbl,n_p = label(pond_d>0.05)
        fp,mr = acc>np.percentile(acc,90), acc>np.percentile(acc,97)
        risk = np.zeros((ny,nx),np.uint8); risk[acc>np.percentile(acc,85)]=1; risk[acc>np.percentile(acc,95)]=2
        cell_m = abs(tf[0])*111320 if crs and crs.is_geographic else abs(tf[0])
        q_map = C*(I_mm/3.6e6)*(cell_m**2)*acc
        sc = gaussian_filter((pond_d>0.05).astype(float)*2+(risk>=1).astype(float),max(3,nx//40))
        lm = (sc==maximum_filter(sc,size=max(8,nx//20)))&(sc>0.3); ly,lx = np.where(lm)
        self.lQ.config(text=f"Ponds: {n_p}"); self.lT.config(text=f"DEM: {ny}×{nx}")
        self.lV.config(text=f"I={I_mm:.0f}mm/hr"); self.lC.config(text=f"C={C:.2f} | Crit:{(risk==2).sum()}")
        self.lR.config(text=f"Inlets: {len(lx)}",fg=ACC2)
        fig = self.figs[3]; fig.clear(); fig.patch.set_facecolor(BG)
        fig.suptitle(f"Drainage | {os.path.basename(path)} | C={C:.2f} I={I_mm:.0f}mm/hr | Ponds:{n_p} Inlets:{len(lx)}",
                     color=GLD,fontsize=8.5,y=0.995)
        gs = fig.add_gridspec(2,3,hspace=0.38,wspace=0.30,left=0.04,right=0.97,top=0.95,bottom=0.04)
        P = [fig.add_subplot(gs[r,c]) for r in range(2) for c in range(3)]
        for ax,tt in zip(P,["① Elevation","② Flow Acc","③ Ponding(cm)","④ Risk","⑤ Q=CIA","⑥ Decision"]):
            ax.set_facecolor(PNL); [s.set_edgecolor(BRD) for s in ax.spines.values()]
            ax.tick_params(colors=MUT,labelsize=5); ax.set_title(tt,color=GLD,fontsize=7); ax.set_xticks([]); ax.set_yticks([])
        def _cb(im,ax,l): c=fig.colorbar(im,ax=ax,fraction=0.035,pad=0.02); c.set_label(l,color=MUT,fontsize=5); c.ax.tick_params(colors=MUT,labelsize=4)
        _cb(P[0].imshow(elev,cmap="terrain",origin="lower",aspect="auto"),P[0],"m")
        _cb(P[1].imshow(np.log1p(acc),cmap="Blues",origin="lower",aspect="auto"),P[1],"log")
        P[1].contour(fp.astype(float),levels=[0.5],colors=[ACC],linewidths=0.6)
        P[2].imshow(elev,cmap="Greys_r",origin="lower",aspect="auto",alpha=0.4)
        _cb(P[2].imshow(np.ma.masked_where(pond_d<0.05,pond_d*100),cmap="Blues",origin="lower",aspect="auto",vmin=5),P[2],"cm")
        rr = np.zeros((*risk.shape,4)); rr[risk==1]=mcolors.to_rgba("#f39c12",0.75); rr[risk==2]=mcolors.to_rgba(WRN,0.9)
        P[3].imshow(elev,cmap="Greys_r",origin="lower",aspect="auto",alpha=0.4); P[3].imshow(rr,origin="lower",aspect="auto")
        _cb(P[4].imshow(np.log1p(q_map),cmap="YlOrRd",origin="lower",aspect="auto"),P[4],"log(Q)")
        P[5].imshow(elev,cmap="terrain",origin="lower",aspect="auto",alpha=0.55)
        P[5].contour(fp.astype(float),levels=[0.5],colors=[ACC],linewidths=0.8)
        P[5].contour(mr.astype(float),levels=[0.5],colors=["#00bfff"],linewidths=1.4)
        pr = np.zeros((*pond_d.shape,4)); pr[pond_d>0.05]=mcolors.to_rgba("#3498db",0.7)
        P[5].imshow(pr,origin="lower",aspect="auto"); P[5].imshow(rr,origin="lower",aspect="auto")
        if len(lx): P[5].scatter(lx,ly,marker="D",s=15,color="#2ecc71",edgecolors="white",linewidths=0.4,zorder=9)
        P[5].legend(handles=[Line2D([0],[0],color=ACC,lw=1,label="Flow"),Line2D([0],[0],color="#00bfff",lw=1.4,label="Channels"),
            Patch(fc="#3498db",alpha=0.7,label="Ponding"),Patch(fc=WRN,alpha=0.9,label="Critical"),
            Line2D([0],[0],marker="D",color="w",ms=3,markerfacecolor="#2ecc71",label=f"Inlets({len(lx)})")],
            fontsize=4.5,labelcolor="white",facecolor=PNL,edgecolor=BRD,loc="upper right",framealpha=0.92)
        self.canvs[3].draw(); self.nb.select(3)

if __name__ == "__main__":
    root = tk.Tk(); App(root); root.mainloop()
