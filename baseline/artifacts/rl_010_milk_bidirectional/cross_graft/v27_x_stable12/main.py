"""v27 current-meta midgame reset for Kaggriculture.

Both seats use one coherent fit-only public route selected after the HIRE4
opening became the dominant Top-30 prior. Runtime feedback remains limited to
actor-local WEED repair and ordering route-existing SELL slots by official
price impact plus bounded current Town demand. Opponent identity is unused.
"""
import base64
import copy
import json
import math
import zlib


_LEGACY_ACTIONS = json.loads(zlib.decompress(base64.b85decode(
    (
    'c-rk<O>Y}nlKd|^^I(#)Z0}8NbEbt+TZSwzG20Lt4a_VSSj--J_qN#ozOqEJij|R(k@;Rxvd1@CCad1}%Z!YS{Plm&{{8nq{_*!e'
    '&i>`svrm_wKcC$%&i>=~|N7g1Km6h0<3E1?<3IoYKM$XOJ^T6UcJuJR^uteI{`%YH$E#m1ug?}|?{Btei>3MV=bty5PiKqs{eOJk'
    'Y(6~vdHeI`^6qT$dh+LAHrF>FM}Piwd-LJT`@8WE?*DIb)QhY4fBEuh^!`JCem&c6KHohy^zdQV=h4p&?HhOBd&jO3$8Y&~b9?vm'
    '<3oo}_C33w()a9|sXqIsFIU$eetY=m-'
    'IuQuLLNN%rr!GN%lDhZAkiV(ee>%q96kTxKR(_aX4ZMnpT>)Vz2^9fM{|97x4HG6|Nb%<pr<e3aoP7^|I*QOcVA-TGTCJ4aYNG!Q'
    ')^!^JPs^-eM0SX4^Q(4M4m|d_|G?Ab^{K^Bb-'
    '2goQH*Hhodroqt^N3&@_LCQ_GG+%ls(=(lCG0xK!qG|64E|PaUW|Zdh;AKh>URhqudWVBK$84f}^|E;}v)Wi&dkfu|3N$00i>ybi'
    '*Z_WtJjdh`D7w?A!e@2;+|{_U}u_C877{)KA`HG@20f6JvB3f>wvG#H&^v-'
    'f+q=LA(YfBnGt@sl4vc|kupJ`+E0uD`l&qn+~Pkzo(e_-'
    'GgRDgWtUg~TV1Z~j|9Yf(GOj6ZZdG_b?V`{Y?O=|{`$FkFh2hJy1Swq0qVf0y7k#y>a16dv+>`=IkMfx*Y4RB7PU-cK!pk=ImswH'
    '@F>6NUjcEs)0-Oq(;nVFQ_GSvX3~5EY(r7$N&xb%ek}@c@-'
    '?i+`5iR<F9FJMS39Tu%P{`R?{|`_tz3_OEA)b@4KseCU2D_PQRQ=b~)AGWYK3Xr@|wBDrD<092N*RQ=wtjkCuZ9+76ZYI^-P-'
    '4np?qZe_H4j9-'
    'oJ3?R*5!M;|l8S{iERWLjhK9NLcPBH`Gd(mT#M%oJOt9&)wFid_KouJ~0o_`zz8??i^N=>XpvmJUXW}$m`s42J<u121KIst_+id*'
    'jqKoD@(qC8a<?X)~E-'
    ')~d<eDgh4ha(nJQM`dDo*m$#g>}0JK(kD{2Y^@yWEG?NgW@)jXiK1{m#emnrsK+o*!;yWkPh1+=fHtS&~9wl|Fy}SNHzpe)DJy*W'
    '7%E+@yQ|=WTSQx~DX~{#O~}paF6~HbU%z#qMcrDYY9M&ut+f2lE8x0wIU_c0+t>dxWsQqwKFlb+qF{*kc1k<E-'
    '|<+6u#cdAO3EKD2G3>0^7qIu0sw0;D_PinEwNiz}|8Xf4+w>t=mS6}aTY4`~Y17~;{1X5cwNg>@cOP>gl(nwMo9j2)6w*aghp2>n'
    'SPhv)*L<nO;k_<=z{4A<y2@WAuz4uG_fPEc&4ZU*QPq!SGNwleO_kV)7c_Az)Ngb(^~d-'
    'Ew+2gJS_Jjt7z>)lBWPVn{K{d>50KAgq2okbhatBD)B>KTOIKQVJ~=r$?$LJ!Y5EH?eg#Az8GYhba7QIOzed)^F4Hc2c{<*KwXqk'
    '=1db9=-'
    'PcnN)Z+%rqMX_OSR6hN~oq9BhyUgMRh#Zlx4&YJAfcRHi8HSV*3Au2h<(}Le5Kg|mCp!cdiU6+>XAQ4D!^UaUBY)WAMIiurr!yfs'
    'S)ipi*=?E+^-J-=%3zpaMWq}Tr9Q&r`XArN&n=`haI+Kj|uR>EJubr-hQv{ycS9%lG<eh-^9J~*%=#6S002-'
    'B9vtSAM#&g2tQCP?ToQv>A=3dlsSQ25?&O0=W9r#rCNSqjlcwyM6zs4TaGN+vTRG^eet^|Q_?Sm(?!@^1W(uFCtN4o|+_F!;-'
    'xcc+tONBSagg<2Cqp!CFp~1cYZx^qTN`#n!yo=ca({A?Zyu`C{1bc8~db&*u?5gyPDRYO);1R@t<q9HWhic_IR4d))m0X$OzP4l3'
    'l2YruZhU?p23EKDSN9##b5-{#<47tgWY`6=Ce$nrP>3Th7!3$nShYpL{T<Uv$Y+{)aR^?IG3o@`nnZfAX5-'
    'W>>J&W~ZarrdGWmH5Wfhplt~Z=}S1^l?wRN<l71|uI|AEOmyv>Zg>pRDbhs7sC8}cw2ETJ^<LbQd$b2eN=z_s&F1M1;sl0wNz<+C'
    '4Mtm7~o%6h7vPGIyAC!_~cb9zHB0zzzyiOf`Z7$VeQ9^>GgVi-@Y+Kc-'
    '<&jW}I^|k&_X!QSbb^VvGqkv;3%U9M9c+(n3X)^Iwjs?U`eC8iS*1iCsOZGaZkYr3%ffXQ`$2ohsJC^{p(cW~GHm$REBtovDo3R7'
    'Qr9c*moWS@5uW(8x%a)wxbqF-iK@qHbHZHxK5NbmX3%ame8loIvX@m{~h;mfYVF?vH42{JYzS%M^IV4=PO}#w41DI}PJ$BeAYzUB'
    '35iLP851rEExpy2aLBrlk3ftHS1jwZmgR@O=EpaEutFul2(R>&`n)yTha;D(2pMgD3-'
    'apa0nJGDJ5$NB{_u9E*470q5QM4|c=)yp0?CfI9HV9o9WyeC~Z#p<g;LOt<BpbU0Ar<zeF2sFD1WS8k(0HFx-'
    'kK+1kD{$>hY$`z*7`4U=xR8Y?V9!8`~-H#C~WMCwT-OC@X*|pV>eaq4q{<qm)pEUkibMm^)1poQsma=g~j~xo+jvkfV$i601MMe)'
    'd2*n3dXEj0Cr1&)<gYyQoLX$yc%fs2Nq9wG5|9eb(YT0WF#uu%&<>@aE;7o-'
    '%WLxP8nBXGRZ_a5^oklycrhGE*AwlK2Dy~CQeP+nJE{Z_2P3Q-n!ffg)9l=-'
    '5pAfNXU{Iv26NnR0Y&Jh?#<rNs{pYA%5va4<TftILAr<oUC&2wh*S0VL@<U9=3uHG~g~p%YHm#7w-'
    'CgcY^gzLWz@af`MWezBp2it2H`$xuxThF2?`&v<Ud@Rt{<bnD%+hLB9gHOoJwpih+#_QM<Qj@g#N$%8~>t0juO|Qf7f3ju!$RWcy'
    'TKnWDW=h&HFYk<h2?x5$)kgfo`81ax*kB`W*NEx|ZA9ciFPLgguOI}OR}3BHL0WC*@dXhci=JZ~VG@@pxF+cZLHs3{pRE!(o0Nvd'
    'IPCx8wZCV+0Ra-Rr}HUmYXJHDi(btyE8GIl4h65V;%9n@hR9J%o?fSykY^RaknIfI9-'
    'TRXS8WsCM2SXV_w)sO@%P4Obws7)fRFvZMF95?cb@J5Bx%w{*<M<Uh{Z+F|U;n(}zJDV6lIrik<s*O;HYUOqV<G-'
    '{C_e7|GC)tA6S(Eq=il0gg-r^7inQ2VLT?y<j3QH#fEN<P?V!Cxci!$VNL<FWjZ0jWB!)oVL?%uiGo?z8IW3wvx&?gw07A@#k-'
    '<Weu*)1t`NKD-aaXuve$em;R+woDka{p(kJI`<L72!%krRNTZ9R`=f(kmk%$&MA+H78FKPzbiq<?f@21hF{H9N<y%KI~y-'
    '?Iv6zXsR84cf1j`5E9&R)qG?xzpm_Zre#p!071g8B&ZUP9}yAvYGNeVyxjipC-'
    't%=GzB_h_Uj256cD~7%tC#LFz+H$wANpDTl!!jiHEF;Owl;9?l3tE_)pw3?<*!mb_M4AW#vk1MQE&B%28jUm9-'
    '(fmCTvPBdCUQMQ8Ka3URLqg_1Nj@!knhU*`HbbL-nl+I><INJF$V;wt;^O>Y?zxm043GHroOys~2&66JE?Z;{W;2t<{(CH&O|=RA'
    'fpZiTCh?z7(rmW}_I7;-+!3)#n*8+`B8_<6h3lhrquTJp^4@>@r?Bb_$irf{TwI(?wIO-'
    'T5(fj0t0^o|y{M%~nk8x5;01sNnQB_^a*f#zY#3A}D*U1m%sm36RTk&)0=NAhr+-'
    'Vt(T7(cXzIEcB{JgR72lZt7|T%82@H4|$Et?>E+SEc<<!zs;E&(T?23$o@Cedq!qlz*B8P*7J>*CMs5nJvu&qq#(ViiQ(X!+hEE)'
    '*i8*VCRx-EP<_-0@g*w>{JoF)AMRy6q9$Hk8cuMj(w{23v!#ZMEnl^6Y#Jbvg!SpWuM|b?&E>66Qltq0CS-'
    'r%NB%(WUAJ4)oZbhW<0Jo2W#D4(6^Z6zKt`cYwSy|rJVfV(#{xXm`B0LMK+?J7$jwbuq;l+iav$xncW$&^F?GKg6vDJ&8XaqhvWN'
    '^1`uD=m0UB99AfTmbSmQv7CYaPOsS;*Q1FT#{(eB-#L3F`#4s0&(y+;^6Fit~Pq=5nH_6@+R82HD5&LBdaN^kmz-4?Jo1-'
    'b2KSUSy)QVC*0wXy48v#VLa#L4#C=qd3-WfJL$oZ<(MiObDD1zCP{VEMtTc+;ms?E&hi@y6l__DU-%~DrCsqkv!7~i^-'
    'bn5VE)48V_eYS`2S%Vi%LxGOhJePwi%OV5MDa-H$!L~V^>QvSi4_f9O9ff!y-'
    'M6JQ!G=*y?wm|%4`^%!i=kyR*#&^Zyp*1bEtwuNZd`Xpa@R9q0-'
    '2W>@JJ5>4NYeI$0F@jp&SM|?*P2WfILj+jf4QHY!^5L8v$4(L8MAo36(30Mc64P80*5J#<Z_$I#4uXUE)=0)w}Qj97JMx`64xi^T'
    'N>Ep6F;Os2Q6B6ezN(oz!Tg{&y~kcN0#Q3T?F%*-A-'
    'GxN(rC6m=@~b68n~SL``ugh2Tg5Ge5iTP3uLteb3OAx)04UEC7QRbY<I<*SaPq|0*~COs~kG|bI66*<jIn2$ZpnyZ@%$M)f~LJxQ'
    'C{!gfp4uADps482qH6k}Xb;$WQTv1;<5fV<Y#{}p9B?OMOGFLqA$$C?%AXE=9C&VfB6jv=6$CKI8-XJpw$Gq<dFb`;YQ7Er-'
    '4KJzOyv8bKS#Ui5lF(d{s!t2C4(c4B_#t+1DyTv$Q9nwuro|N4c_3t@)gS;zqgRfW<2C7SG)8c!sL)6$duWGdYdj>5x>hWPBE4d='
    'aafa}tzsIY&zT8O!unZ-fk<3B&SYl7V@;udEnl6T#DxU~0B=0F;JVUT0+c1yjU*yeGmbd~2yIf%BVFt%DkW=o3s@{7Lp{gXKteo#'
    'o1R`|OlXU2e?c;Xi7#%IYS3=1D3~QlJ1mw=+s~d%MGR}0v=`4N7!ROaDRdH{e#t58aSt_6qEk96%}rFoACb4!Qd9}=QwaR!NP<t0'
    'uy;;!!MB%*Jq5jHMz>nj!{w}lPN}rnYSo?}Q_QGU1fT-'
    'vWxIJk(PSVfB#`^IV!*ymAPw-1(A@2zOx;CCK$mPnOR2459N_)B#vElhk%Zk-Ezm8c98lM}fv6)z8HI%@(by9_GDb&WCydB4>sys'
    'Ol3<EmJqoZ@28-'
    'G8KG4Ujy_ZNND!+zu5(=seia!Y68BP_sG++l~AcQUTKmwA{aAq1aVrL|lxjIyu44E}gdCm=n=B;Be@+~{~>Ji(O9HA2|B)u<LV`N'
    '&HZ#S`_E21~7#Z+R>*EUx~@rq-'
    '}G7Myy^9&KpHU=UO3SEPLzbBbvWhpf8zK1OMML#EYZk2CVyxtHoDyrF#P~E~4+iv<0CK|=7vN&;fdWXI0bJE>}a9X5qtUGx?phL2'
    'xcP_S2p>-'
    '5<K_f&|%(^Va3A1YYI8Q)vpn0*Mt_u5(S}@D}&E=!Fr1j@@_wzh|V0g#1)8}AeG57>boGMr~lM+Ao1X2Qt$=F{PWktN~1)rXuCZ)'
    '1cj3-'
    'ZJuq|rIf{URTYC9H3)!b8R<Ylr=g)79SPf#cql0BLM3k0NSy7OqCNi~&@)dv<0I38cb@lipE8pIcC9cw5`Ae;w?>)h0`Kx)Ez>5P'
    'Oc4p_r{D+=4JL`Epgof0djwcV8JaqTF?T*`-VDwV?S1#SR<6qYrm^CwkvPr&aezpSll_cxJ-GD}3>7t|}|pU{)d&Ejy9N^%kGIm&'
    '}8&`e6WVMd++z(CTRS*f|tsRU2Ul4ivqfT_fOF&phbmzYCywlBp?F$<n2-_A~eS-'
    'L`!iJneFvh0xyyB`%L(YGjGC}pKMn5yEmFvW=a95yK+a$*Flu%*?Yucc*&J|Y?9e301}whP3DLKXp)3(QrD!(mDxtpJ~t#q#g;ND'
    '5Cqq6f*rt^MAQ9cikPBaBuqV~gg;kqou=(FgK5#B2!ka`L&2717ShJxNYvayHSzwcE@VV2F`6cpC-'
    'A6^qLWX*g9KRcZcpvxa8X1pKAiyab`nB6kd!&(cUK_j_h~1avyA*Ggk*Qn({E3G6OSZEIQ+qygg2Y3E8eEGL}OObu3A(1K7xYRa1'
    'eXcPOB;oh;ls+?|Te8uOKLHp;VJb`SA&PqKq<w&IEi172OAHC;gq^2~Ax13*`D8HpkF%1r#y)J6C%n=XGSKnZs5({uwDl}WZWTHa'
    '2c6H{<KJ-W)(4BeWG!BeSr9yVZTA|;ZRmDZmhKz&C8JAfttSyNi4k!EO`sO1@8Qm)6zQB-'
    '6bDuPWtE<Ed$)}FDDAYEjl_n~Dg4Dc8nlQjWw|E@W1W?>T$Luap^Ugb=`iRHWD!&B&G{ig22!gdi47c4JFNK=1>?(`F!VqqRO1}O'
    '{6^pkein;igvv)~C=4=dxOr}IGr08u$twN2jVo3sqB9g(5qTW4nunZ#B20B5PqXKp`)IZ9J9Vt1yF_Gvn`MH(KIo?XnOE}dLzh7H'
    'zlBq|vUFD?Cpg+@=aJrobP4i^eAn2Qv$csG?uws+sQTTOcg^~7RW3awJh2$}yw@a>E$S3zhD}nS9_e3)i_Dp#>QJtGze{#(L3+iq'
    '2fzsffCfYtCFrPyzQ|IA+sjco4rLzZv!6(1zQG0j_`5>Z5k@F5l)-'
    'TTMv)vDK6OX<iC=VgneA*ufz^!U9=fU81sDNA>e2j5_N9@3^+SE?s<51lRR&SWE3e!rhb*R`!u$>&9>qYD9gNEoOoA@_lT3@GfPK'
    '>d}3pi0JPttw@GL|<<tE3F=DUA9hWT5f4AIUE-pkkDu(;H`>Fr2iZz7)5*5~+jyLAAn3r9A1NAsb0Ynni4ivQXStQ#v$!dr#|Pu7'
    'nFS1P)8H-'
    'gV&3;GyMKSCt?s<wvW!CqC=e@{aLJG6z!Vru_8f>iWZP4|dYs7oVgiCbf|cARUMH0|#X}U0=T6>};vDu9tOTKnM17q2)PFEM?<)>'
    '|V)E9&!)Ez<2-*lUa(G>yt7xft9<AtQ4LYjsBNIPn8Pr%S2cqaOtBW0b)h8gf>(n@dlobhxHzXf&k>R-v-'
    '2DGI;?(*N(i*SFzy>*M(ok4?FVYDu{~>&dMXeYwIv)An!4f=F65k=as*fpfXjC3`z}gPj-vznRO`|7661%oeB#M*>-'
    'v@?~{~jvrY9hAdb8xJ5VbjmloSvpIeY`x%v*scTF_8b^oN(6ynjydE=N-'
    'oH*g*OPxxM5&57bsMJLcra(|*1;rxDSwVfL&NtJGQtQ&#eo{)x;bSF8!<z4cPDN=Kag$k5yOQ9Rwv7Y^%nwNh6<A0fTDLT5<^}5#'
    '=1O&*26-'
    '+yWmC##N>oT~tFEjeqS?|do;Q$dNTt|hP!CDcj?L8gAz6n(14zZL%K8K2mtTPS>_fH+64eBGS}<LQEX$K%8I_y<>9C7*-'
    'x7E`Eo&yHHyt^L%M$qrUyg!?yXgAnwihCE8C55GQxR9NVgBVa9_vxbRU!f3A{f*13TARnit`DS&BMx8{({VEN^5f6r8SsxEkQd@Q'
    'MlaGy8b7~e4EKZe_eIZ;nObE^4u|9{-'
    'kA|n2|^R(#i3+#+6tUN+J<ZMn4~xZ>F%6TDUW!1<q583Jte@H&Pq{ea*yaPglK)4td@>S1n@3p-%X`5uHGhX`><dkd-'
    'R&*{c+S;+9M$Nn&oq#$^{o201vq!f<}TP3==5833_~VWI)WcD50qD+tiSR+4{DmVYMHrLMJyB_F@Q>enl37=*Bb@rGe>dcG(*Tk<'
    '}$>!K!NMI}`%5sZSLh7v=Q*#zoPehu0w6PBP{K03wPqBb7CS4tjYLbiPiQh!q7{y6w97EV{smqk5%L{18t(hF8DVy}Q7kYmLcZJ('
    ';dlqXpM@s+rXX0);6sCWw5^Z^T?0E~5%Fq}M6qoS24ze<si*wOnI)hiPKU;=PUB4IWSE8507PQ~Qq1wxob?bvxra-'
    '^6t@AOMW*p1P_f{~?CSJXN3fV~v8H>d8U>uUuMVh15-'
    '_~miOl|4+KWVpT;<guxOo)0gsixu@~>Tj<EGBF#>BLImqQFCXEQI9|~8kzi8PnblidQZQtsTZ9%VFVMHqFkzIE=9g}B2R=ZhUnUt'
    'G_R;8bQ-A93hr^tk#G(cTA{7BN_$+pTlpLntuFAR!UgGD^ch93t&kC-oiXX?0x<WON=}HlZF(r<N@`BAzOq`!kcscMBrKwh#L^-'
    'nLsiTbXkq=KPyR{{Rg|+z)iU2t^fAL^#NN3k(^Tu&NeiK(A*)BL)9H#+2Vx#M-%`Om0YI7Igaw%}W8?YVeL_)R!q;l-'
    'N{GZxJhWnoca*Ma1ZybB%Xe$4S%GG})G$Si%d7K}6jed%&Hx{Wd9o>Z)DB`|m{0Q5OHZ~1$Wx!YnO%uu3!<Ej9TdRyBGW{A&YNB='
    'G$^%~VZ%KMR%r6pORmE+D(Lqt*vHK3;*ApG=bl)f!n9hKD-!kQQHioTwCGw>V<q;5$eR*d>0a=sBt)?kDow>)r9_G5-'
    '%?lF*$N^pIwdNuu^1sJed$k}k)!9qGFFTf@$3N6RWeVI=vQR)s7fjQp*zND;jAUGN?bppSw2}&AlbQ-'
    'N3%#uRi?5jR;)l4Zwi~QTo_K+0CC+LMS)y#5b8S{Vy1NM=c?VQXI|@eCMgqgVL|4oGXjyXkJ^K=#uH`}u>wE)5%-'
    '}IJCHE7K*S=@X(k&yhuzRsPH;bwLdGiA6_V5v^eT!)$~6T85=!fag$GYSY0Z{)Ru~%$%E}`y^;3vZi6r?ls+yNWXN(|6BXJ0{HWH'
    '|z{c>je_;oc#N>;;1X4Q0Q-FzQrK!z>=GW5zIL$ALlIp&upibGj3lz-YqCPyX7NG0DGlja#@bF`r~l*d!ZvuP?il0_xeOn%||IyC'
    '=_g$V_E=0->jqS>Ev)SFCKYgAq^v{(X082$#l;r@YoALGn1cn!+d06Lvm$fufel}!{_)|FLQ2wZk&Xw7z!+*j-tRrO$<3nf-'
    '7d@@B5NGVt)Y6PiT&y7h#O0`ndtoWj`YFxaCNiy;rxp;ZdRiaC>wI7P2D595=lsGK+O{&TTuKuy%l-mQj6fz-'
    '~g~cQohASfcK`xe2wk>A9M&^+pxM7R~UK1)21YoQmg4NQHCOnlC!cuEH!&}>U2=;JQ>O58HO5`OuZ3H57bWAPEA3KD;Bxo5$cnpG'
    'QoZjW{1~&FNJt^hG^x0hnlvY?k35>#Pkn;2$fICfzPUragkn)<=30arM#3mt_$@wlK=o5R?Bt=5@C@Lkf+SH7e<$Qy5Wh`C%xJyU'
    '4?4$skX#z!5WTIQeCn*=GUM`s4ueltY^fr)$(jxQ~_!eaW?ZOU->L}^7ZmI5#4owm`Ox(2;b!TO9I)=3Ut*NS^ZD(A-DRF_6>P-'
    'Z(wr1n-VnJkMA~%VIJ9-'
    'f@BuqIb@#t%z0uKr$zI7H!o7#8qBF?F0QGlX;y|spI_9&n#ZpnF1Br8Q1=xuDXXEKJvGH6VV(Nm3ys=&OfR)O*O09K#Z79vxV!B{'
    '`^=CK;xOvo$L*<d&9MZ&Xp%TW&^E+IH&d(;S=)TBsQ7Ll^O#%}m|XO~z?bs~*yg%tod<oq?w8ZsgwFEg9S!$rzXi>wb=Xc2||(9s'
    'iAX`~621z<jwEMT6kqEOsl&6Kk#`HkZ{EPd*yP)}TqF{xZ_PF~d;oh(45XEN~Bl_K0FgbVfbbLJDo%a^IpDssR<ibjG;O*!h67R4'
    'N}z7wc$ymG!J_klLE%4D*Y6RTX+97ru4Q_1lv`HZ$f!1RPdjB=e=QrfprDpC=-'
    '*rqDk)Qa>r+<+9A&{9Wv8RRs{7(O{H3B1uXA2LIiYZxavw3Q=cp+yoyCI!U(sybvFyj8b>;*5eC!dihJURB4bOl`dky~+ISfhYj0'
    'd=p>$DbJ)SqWq%~LL;Wv60?P0Q}?w|b^`i}xeD>bY^3oY-Ubyw)wIBfO;f2#$i>E3L*ik}#@wlS$Gu)=fyTo^E1w6v?NnNI+xUo<'
    'q$q)yk&2_~=<55p5t2jvX%OC7!J^((WO!EVG(_2@lsD?14fREoNNtVUo@kWyOa<>z(0Wz#15!;B0%`mtmrm8<#;cGlRr+Ug<d~{n'
    'Fc^%TQ49UTWgq#9kqbqCLw}rFaOmU_6bazBbYL1d=2WGC73?_pD;gvnM>BTd{Kb8((VV3hq9iDT@v(=MPMoLu07eqHh2+EZ+Z3k0'
    '5IC&6#Q3*DHxkW)POe)fm3Cs4>4aTlrt<Amv#5im^r5X+>kC!-v2XbU*vq)NZys88|NntQ0?7'
    )
)).decode("utf-8"))
_REBALANCE_ACTIONS = _LEGACY_ACTIONS
_PRICE_FLOOR = 1
_DEMAND_ALPHA = 0.25
_MARKET_PARAMS = {
    "WHEAT": (25, 10000, 400, "sqrt", 0.8, "log", 0.2),
    "CARROT": (35, 10000, 450, "log", 0.2, "sqrt", 0.7),
    "TOMATO": (60, 10000, 200, "linear", 0.4, "sqrt", 0.6),
    "STRAWBERRY": (120, 10000, 100, "sqrt", 0.7, "linear", 1.6),
    "MELON": (250, 10000, 300, "log", 0.2, "sq", 3.6),
    "EGG": (50, 10000, 332, "linear", 0.4, "log", 0.2),
    "MILK": (160, 10000, 122, "sqrt", 0.6, "linear", 1.6),
    "WOOL": (200, 10000, 105, "log", 0.2, "sq", 3.2),
    "FERTILIZER": (100, 10000, 200, "linear", 0.4, "linear", 0.4),
}
_SHOP_PRODUCTS = {
    "BAKERY": ("EGG", "WHEAT"),
    "PIZZA_SHOP": ("MILK", "TOMATO", "WHEAT"),
    "BRUNCH_SPOT": ("EGG", "WHEAT", "STRAWBERRY"),
    "YARN_STORE": ("WOOL",),
    "ICE_CREAM_SHOP": ("STRAWBERRY", "MILK", "WHEAT"),
    "PET_CAFE": ("CARROT",),
    "SMOOTHIE_SHOP": ("STRAWBERRY", "MILK"),
    "FARMERS_MARKET": ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY"),
}
_WEED_STATE = {0: {}, 1: {}}
_WEED_REPLAY_STEPS = 8


def _get(value, key, default=None):
    if isinstance(value, dict):
        return value.get(key, default)
    getter = getattr(value, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(value, key, default)


def _regime(configuration):
    interval = int(_get(configuration, "townCenterSellInterval", 12) or 12)
    return "rebalance" if interval >= 24 else "legacy"


def _copy_action(action):
    action = copy.deepcopy(action or {})
    return {
        "farmer": list(action.get("farmer") or ["PASS"]),
        "hands": [list(order or ["PASS"]) for order in (action.get("hands") or [])],
        "market": [list(order) for order in (action.get("market") or [])],
    }


def _seat(obs):
    return 1 if int(_get(obs, "player", 0) or 0) == 1 else 0


def _farm(obs, seat):
    farms = list(_get(obs, "farms", []) or [])
    return farms[seat] if seat < len(farms) else {}


def _align_hands(action, obs):
    action = _copy_action(action)
    expected = len(_get(_farm(obs, _seat(obs)), "hands", []) or [])
    hands = list(action.get("hands") or [])
    if len(hands) < expected:
        hands.extend([["PASS"] for _ in range(expected - len(hands))])
    action["hands"] = [list(order or ["PASS"]) for order in hands[:expected]]
    return action


def _tile_at(farm, position):
    try:
        x, y = int(position[0]), int(position[1])
        return (_get(farm, "tiles", []) or [])[y][x]
    except (IndexError, TypeError, ValueError):
        return "LOCKED"


def _trace_actor_action(actions, step, actor):
    trace = actions[min(max(int(step), 0), len(actions) - 1)] or {}
    if actor == "farmer":
        return list(trace.get("farmer") or ["PASS"])
    hands = trace.get("hands", []) or []
    return list(hands[actor] if actor < len(hands) else ["PASS"])


def _weed_repair_action(obs, action, actions, step):
    action = _align_hands(action, obs)
    seat = _seat(obs)
    game = _WEED_STATE[seat]
    if step == 0 or step < game.get("last_step", -1):
        game = {"last_step": step, "active": {}}
        _WEED_STATE[seat] = game
    game["last_step"] = step
    farm = _farm(obs, seat)
    positions = [_get(farm, "farmer"), *list(_get(farm, "hands", []) or [])]
    unit_actions = [action.get("farmer", ["PASS"]), *list(action.get("hands") or [])]
    active = game["active"]

    for actor, transaction in list(active.items()):
        index = 0 if actor == "farmer" else int(actor) + 1
        if index >= len(unit_actions):
            active.pop(actor, None)
            continue
        age = step - transaction["start"]
        if age == 1:
            unit_actions[index] = list(transaction["intended"])
        elif 2 <= age <= 1 + _WEED_REPLAY_STEPS:
            unit_actions[index] = _trace_actor_action(actions, step - 1, actor)
        else:
            active.pop(actor, None)

    for index, (position, intended) in enumerate(zip(positions, unit_actions)):
        actor = "farmer" if index == 0 else index - 1
        if actor in active or not isinstance(intended, list) or not intended:
            continue
        if intended[0] not in ("BUILD_PASTURE", "PLANT"):
            continue
        tile = _tile_at(farm, position)
        if not isinstance(tile, dict) or tile.get("kind") != "WEED":
            continue
        active[actor] = {"start": step, "intended": list(intended)}
        unit_actions[index] = ["DIG"]

    action["farmer"] = unit_actions[0] if unit_actions else ["PASS"]
    action["hands"] = unit_actions[1:]
    return _align_hands(action, obs)


def _shape(name, value):
    value = max(0.0, float(value))
    if name == "linear":
        return value
    if name == "sq":
        return value * value
    if name == "sqrt":
        return math.sqrt(value)
    if name == "log":
        return math.log1p(value)
    if name == "log10":
        return math.log10(1.0 + value)
    raise ValueError(name)


def _market_price(item, inventory):
    base, equilibrium, scale, below_func, below_target, above_func, above_target = (
        _MARKET_PARAMS[item]
    )
    if inventory < equilibrium:
        amplitude = below_target * base / _shape(below_func, scale)
        price = base + amplitude * _shape(below_func, equilibrium - inventory)
    else:
        amplitude = above_target * base / _shape(above_func, scale)
        price = base - amplitude * _shape(above_func, inventory - equilibrium)
    return max(_PRICE_FLOOR, int(round(price)))


def _is_sell(order):
    return (
        isinstance(order, (list, tuple))
        and len(order) >= 3
        and order[0] == "SELL"
        and order[1] in _MARKET_PARAMS
    )


def _impact_score(obs, order):
    if not _is_sell(order):
        return float("-inf")
    item = str(order[1])
    try:
        quantity = max(0, int(order[2]))
    except (TypeError, ValueError):
        return 0.0
    market = _get(obs, "market", {}) or {}
    inventory = _get(market, "inventory", {}) or {}
    prices = _get(market, "prices", {}) or {}
    current_inventory = int(_get(inventory, item, 10000) or 0)
    current_quote = float(
        _get(prices, item, _market_price(item, current_inventory)) or 0
    )
    later_quote = float(_market_price(item, current_inventory + quantity))
    return float(quantity) * max(0.0, current_quote - later_quote)


def _demand_per_day(obs, configuration, item):
    town = _get(obs, "town", {}) or {}
    shops = list(_get(town, "unlocked_shops", []) or [])
    turns_per_day = int(_get(configuration, "turnsPerDay", 24) or 24)
    shop_interval = max(
        1, int(_get(configuration, "townShopSellInterval", 4) or 4)
    )
    demand = 0.0
    for shop in shops:
        products = _SHOP_PRODUCTS.get(shop, ())
        if item in products:
            demand += (turns_per_day / shop_interval) * (
                2 if len(products) == 1 else 1
            )
    regime = _regime(configuration)
    if item != "FERTILIZER":
        center_default = 24 if regime == "rebalance" else 12
        center_interval = max(
            1,
            int(
                _get(configuration, "townCenterSellInterval", center_default)
                or center_default
            ),
        )
        day = int(_get(obs, "day", int(_get(obs, "step", 0) or 0) // 24) or 0)
        multiplier = (
            1
            if regime == "rebalance"
            else (4 if day >= 20 else 2 if day >= 10 else 1)
        )
        demand += (turns_per_day / center_interval) * multiplier
    return demand


def _order_score(obs, configuration, order):
    score = _impact_score(obs, order)
    if _regime(configuration) != "rebalance" or score <= 0 or not _is_sell(order):
        return score
    item = str(order[1])
    quantity = max(0, int(order[2]))
    market = _get(obs, "market", {}) or {}
    inventory = _get(market, "inventory", {}) or {}
    current_inventory = int(_get(inventory, item, 10000) or 0)
    demand = max(0.25, _demand_per_day(obs, configuration, item))
    excess = max(0.0, current_inventory + quantity - 10000)
    urgency = min(1.0, (excess / demand) / 10.0)
    return score * (1.0 + _DEMAND_ALPHA * urgency)


def _rank_sell_slots(obs, action, configuration):
    action = _copy_action(action)
    market = list(action.get("market") or [])
    rows = [
        (_order_score(obs, configuration, order), -index, list(order))
        for index, order in enumerate(market)
        if _is_sell(order)
    ]
    if len(rows) < 2:
        return action
    rows.sort(reverse=True)
    ranked = iter(row[2] for row in rows)
    action["market"] = [next(ranked) if _is_sell(order) else order for order in market]
    return action


def agent(obs, configuration=None):
    try:
        actions = (
            _REBALANCE_ACTIONS
            if _regime(configuration) == "rebalance"
            else _LEGACY_ACTIONS
        )
        step = min(max(0, int(_get(obs, "step", 0) or 0)), len(actions) - 1)
        action = _weed_repair_action(
            obs, _copy_action(actions[step]), actions, step
        )
        return _align_hands(_rank_sell_slots(obs, action, configuration), obs)
    except Exception:
        farm = _farm(obs, _seat(obs))
        return {
            "farmer": ["PASS"],
            "hands": [["PASS"] for _ in (_get(farm, "hands", []) or [])],
            "market": [],
        }


def _kaggle_submission_entrypoint(obs, configuration=None):
    return agent(obs, configuration)

# Cross-graft: preserve mechanism v27, replace only frozen route stable12.
import base64 as _cross_graft_base64
import copy as _cross_graft_copy
import json as _cross_graft_json
import zlib as _cross_graft_zlib

_CROSS_GRAFT_ROUTE = _cross_graft_json.loads(_cross_graft_zlib.decompress(
    _cross_graft_base64.b85decode('c-rk<O>Y}nlKd|^^B{idV{aOp(`}5}GGz4<vxd-UU}v$wV)oFxx5fPTb&F(?RT&u>nXgKsdwioQv+8}n%*e>dFaLM(AHV+g_rLvi@h@L4{`B*Q`}e<my8Ha}^W)~?d3N!izy9~X|IgRIeEs<MUw`|rzy0sm&%gh-zy5Of@xxDd_ZPE^cMsc(+46Zd`1<pYo6V<-+5GT_&zsHruV4SPxqtY0F}ogoz5Q`>_w@CDUmm{y^!V`p=XXyhe|q`Jj~`yzb@=f7kJ)MSKR!G@{dn5!FBjX*=ckt+?Elv9i29VnS6?nZeR%hmpFbVCeKCLea`%$jqlZKO^%eJb?>754oU|OiXYl;5KmB+dGp7sPkaQnxr^q|DcTbzgs1Np=aVDaB%I5CS_Cc5J_;tP8ucV-VhX?LfYVYmfUEtZX;~0Ip`26tmaOLcGMt+!&Pd!d%urk4558tyJQMgk6{QKs_gUqHgKcI&{eYtpdcUUi7XU%->|GgW>XL~f+v*S54xW_3&!@L`l)PUN3)?Y1l6uMvRhNIF&GU~^!b`uQFguBn5(+jlUY;SiCzcFu;r_m1ean(3Gv+L;be?~neLwQ^)$AR<rN@Yy#<BpSdx<$f4O^n@f`d*$AjMfM%vU`vHaB_fFM$dn6Va)cw)t;YOV43F*KD@zWhdXP<EyaO}7n^bW1#S)&(k5UkB(Ht@QhWBQ_DSkB;f3<<;r@Q}?&+65ZyujM+<*AD;TlJO&$T-2DcD(K*wZERr*QTDbH8VB&{gh)+za<dHu=%vP;dRw&I>q$-|XFy%uh1M=;zJ-?>_a(4xcUR=cwm|SvP&|y5p<o+6g?1&nwlQAl(ZOwCGIjY%F)6;`X6~7QHrQ<=5w0!4INg0ne;^D0?fDx`W08OJ2vF4Q71j(rpJEh@y#o@~Lp7PuQIJ9A=K+S&R;H%39o^ge6Z~lW=9g9Rx9@!gD(0?DZex-I6^#?^PZ$yqXs7)jRGL3oi22%OLK)?=Dw+x62nX>)MHH?_5O+Ogoc3e%cgph#|lYf$Q+~i|R-OL&BS~pSiukbY&-VQ5}`AIbFlqQrkCdog5te?rwv*(FE8Ab*awrPI~|N@M(B03_pe2>?RXk=)e$dLB@x|;06I#T+o&M2(3%K0N1m+sT8Z*_T=OSW)me`dM4iEzpc_6de4GYkm-&!DTY_&499dr*U`=ada^~j<2IAa6*PHp@s>bggHeJP+GQ1c*UEF1juf7YRzHHxl~c;TuR9IF`<N{T-zUrmgX9yo-<v*oQVS8-Zg8Y$6B(1o2A-$U*f&Hnw(XP#WeVGw-n7gB9pf92nP`5+Zlu#djFOtY*nRl%?(tuv&+(neOe}CUA^L!4ZuQaDB_>5J<8#96TnA%9gDiwri~-|^^ATh>1W`{L=Z?kar^mbPpEi$=f88Hxc0d--HSlKN)RQ+j)Cce%<FIBM70X(+xkbuAxzi^l2n$tsiVz@reQJUV0gXeTnt+k0#nO(;owz!^l^?&(+TOrG9eN4Cj7_~_Xi_t?c(gaPb;25aV|RiW`9~~b<D31W@x6F%a$3e=nlSez*ER3Ds#?;^xnoLC5c<RhX`kafKQjy#H+9InHjQ1Li&4Ft$f&j#`YW2$S>p2;eor?b-ClN@dECafnXO~wkjdx0ApkeP`0qX40RRh4pkboF0d@g`4htMM&qu0uh<gr0qBTcC3*9-czhUTNybgi5B)T$XpVOx*;UiJ&#_+BasLiJt52yTh+(F}y7C4Eg^F*QR5_iiU3e@mu0Qp(Sg1tP`L14U-@D=DEg3vwNrwXmp26$j;qcussnfm#oSi|<4I#XJ;e6QZx@HLC;I%ko9zlvz1+@j_4>j9Ix<+dsZ`vapET$rtv6M*LuhHhJC*{Rt9y2jr=&K&@{Z}XRo5m7aG0ALjw=?(D-at!cry$;H;E8T~i#~~pwz!%}~q03k>G_}qi0jWIN?mc8icoCy#LsIh101d-;+QY*=yV5ed2>2l`LKda0D|~IjMF1ARZlVei4C+y!KZf)x(_uH;Kd==2KdMH}DV@DcrU$9jN65ja5>%mo29#>=XdorAmwPsAk<S~FCYty@;LVoMl6<qrwg{Gg+B!dd!Lo(MFpiOafA{hIv=%#q@&rc8PR+wMzSj#7_Fdty?|LkrNS5%c6E&ihu&%lX$dobel;7HRA0g(FB*FM*7sBU3!%+(6tqp2n2wE^M#lO3}i=2Z(sR)qv>i)Ny@ywVsZ0*$i6D2}FqwInKd8B?3vZW2)&jQ<t*tx{sJ~pJgn%V1Px{PA?6983+OC(2ScvIbuZJRPAfjKexLG?TenhnV!R_UQEyX-DO9oxCxVmukj%F<J8Lk(b%xD$X5)!?d=oPyXRVAEVoH%PwJ;hb@|b=KfqP-+kZ`Ny0!tzBsC5LFq$<xUKpl2s`+&~8p)j&GXI-wE0-gls{ZNAggr&gbY}sP2zOE}d^nUyb0WLRzVsOWHh+qO;juUzJdl?0+7TEk4};Wxqrx_edeT7l(S3o)O&E(G=}t2SxwZIE>uN5ZG20wVy2E1ck70qX3-WeZ^R81GIrGy7^+Ptz#!gBzqN52~CtHvl5e?1x+%vKE|UX9Q#B+;xIe0<9k6}7q%F1m)N2M4pMMME0d((hXZn)m(>6?OOdgOmh3=)Go5APr|1l`T!4>d*Hz431vp8%(F}wD3mP;E_+pIkpUXO$C!mSo?qmE-vlE63WDy$C!dOeBQoFXp>)UFD#*yftsdkvb1)#xC7|lC=70Y|2+ZtMRQG@y6VLa8S<#a{c9Y_q!S;VfupvPc!yYWz^y&{%Nhmb~Vhl!F^nh$(Qv_LhCu);y5I1M~SH^g5ybLydiwIxU!%LpI#o57)!n@wPAM0<B++ADeqMK^#|FELs5paw)CMxwjI(hd@+AZqwFBO2C|nU6w0g=j|E=|2r-OpI!PvICGf0KYU)7I>Ng?DG-$X<!^qZT8r^Y59j~!ui3p@%BWWJmMjDj{i1Qy8)G#dB{5dBh--%e0sMU!Q3~p-+%bQL8Pi2j2PP(X7H9b%{+poz*IT6P}nUOU#sP*gQ^%r_Y1GA!`yW(6>atkSPxcsAVOjz{h?`^SNK|{0?|OAgu3*IMt|5I{Xu`3ijhZ}+xgZk>$gZpB{cz(N?=T5d(iYaif*v5*$WSR)fDL?b3?OPqN$-2ew8n_3}gco3zj;%{Cc*1L<a*p4&=RkRXUXvRmiZl{JJQrFj9h<ZFP6j5STQ``fcaR5Na4YlthV%Ku*)U^$!bNEiUEjjYpO*XW|t>5wyH07Q!5EjXVGk46`C`GY~W#R8DGSnG9`$MM`c&fH!HHWumf9k3y(JcSJ!~8^yGyU5W@ofs?lbM;7M`j?5S5j*gsu$z1BOC+%`fKtu=yD_x2vyb)H&EIeS?L>+^)B+F%B?qQ=84g^WM4-tW7%gH>GqbD1I_8{ec=){ox6wH+sWye0s57DOtM2|9nA+an(e}iU@WXNs>b#jI)k@RU~R|-DWc;bz^Dk<K3c5n0@+NOkd&|V-$if5i1*NLM}gN=aw2zCY*Kt(?p`cg&5k{Y<-=8$2$wiK)a9LOCe^oj9eieXvi0&~g&U^*t9jIPW}MuFJuRVyOO!rOePnX`FXlqLqIhAPxT(37Z?p(@3$G@9J+!5I~6Jj}+($HV8Kbq;%B9Fx@^xaaXa&aZR@Qljm^H!*FlO_@0DvJXVrdD<nfml9Odx(^gvK+IB|E4X}4ayO#up6RhM`A(KEC+0E@7MrKOyG+T)fB+9t6_D0iR-zA`Zi{Bp%HcuRUZIZ?!tj<{mgPCYHQClL-w2uk4|-Y8i%ENXdj_l(ZVpPHrYEo<ou)w=d0KZaiihtH11M$En&IP9NkVYv&d;V01V-B78;06I$Uv2|9)b8W%G4MPLRU?pnh!#9L;LLTuxpUkKr)ktQ(PH$8Vv`QMfa6eM_~aRixX1DQOe6^co`QXq{0`Bqbw(4P<}Q`908GC5-n%SlMWvv9B56SAwaFD9JxJ(GDs!!ChZZ;cnHoS5rBBScPmg)#Cm7NIT?im<kC%80=^q}3%`hrKTcbr(Z{0*rb-238<2|Jhs??}<pSo<wPzv%$ByH7t|QCQmD&Nwju=Y0(PoLknN96xi8YBu63$*aw}=Ur(3+%L2?h;aq^%ZdFimtm)^c9ELgwnaQfft-6f1SbtyKU<xwv?3wKvgc=Ie3zGrSOc8VVYRKBGsFnJfLOa7D6hMZD&Xi8FeW?fMd|KI+?NNmHj=@2XjnL@!UJ33*PbcS-=2?i+>VB$wdL==#i4`r+6QVB<-Xr-Ea}(OT%We{tComjlLn1i09_Yat26QtPYRgxD4(@I-qi6u61qUYZJaqXU@xeq=Nez?j?pM%-jbiBauL+X2*E4ga7y7mNzvX49})SzlE$27sk#2V2rE(W|F{*dR;4aD%ZCdhD)uatfI}dR$F`^W>1OY<55*vS8N~KS>W8bbS(r-1X}!P96w-i7+Sg2(>c40CVE;yPMMuPVpX`eTRw}YK~$~M`+omQ%O~EdCx^xG4_wl5xHO4T*x#5JAnj?jL%L93=5lTxH$dp>EWllrw3bZfcu-d;?oV2nCx{&1ZwJxGj%|ExjdV8nI5rTwFu&+IrJ8#>W3Pjs7Yg^IoypKL4?;CM3}8Tgnp5lu^U_yohjxjiHw)!6~?o^l2g$?`RN;{%FNo)r^Xb@wS)U=h}7ykAoUFaQmfa)qe{9q1vqL9>0;(&=+?OwAl9q?fhZrrpc4KBW!6Y+pE^R5SQcRL*jVR|mgX{|%LI%A!ul*4&d3m8c!=u}&yYoC2!a7n4q)Th)!)B#`2{2bN&y~&+A!=^0&1oVEdHEI?_>#4g1=Bo77H~~tgI?7!%Hb`gc_!72J%uwnHoc)VuA+vA!R7v6WBR0%N7+J?lrnvH)PGQE;lTC5D=Hrn*zCQ26j>f)%wbzDSqIOi3?l81;(ig9l~-4MKgOVa8}7w6S0*5TEPFzYNH6{okhliY^RGErCA0J<Avz5O19t_%pRQU@gjyLiX><{gB@tqtzmWk!oI9B(2U(<L%=KQiLBh4P>=(d#o7X<e)#XE4)u`D8GIt*JLH!lcdKEtIWd%gY&7wW+>BviDr`xt>h!5*6oQx6;Qj&v*e;#=TT}CQws~Ef>F})O2v|=7RH&d^=CTQ;XU~U$W!-O&jg_m}H3Cx!BXosK04`3XXT)vG@I;}bl;uXaTi{LwsAgN@n_p(cO|9}i%aYZTD?cwvyeu&8f#%{dDXVGW4Mg9TKy`MJv362&5hTIKE=?(g${snBX-FeeM(1%cuY=x~Z2GSB>o0-~>AAu1H4p)}CkPk%7U3Qx5ePuD=o9#FG`&a3Nn{j+myq_giOsKCbw2nf3sC8rT3*y6U3LwUg7rufha!ZmI&xb@v^Gt|i{f1|GM&riBxj-sUv6B&uO6GT=cUaYq{vJ6-%)*sm}boV*g<Atyu}zr1TYrkiq7hqNn}KJ8dNcl6w&xdE|^WS;)h~ca2jqmHuj(+T1J?^he`7Q0?%}rrIBPwy9CxQwsPxBYyl7jC7#95D81hjZM7pK$r@?%uN@HkEp-a+l?x_|%)E2UsGFZJfvG=9*CqI)h+hFkuJJfcyn4peisG<wL4OPErniK83dOEjl1GljG$FD>%P;}=nZ$<+$PSH9Z`IVxioe|U%yGC(l%2$kYb>zK)>JqX%<H8H8>5d=HJ%E_(!lD8W*+Dhso0swej-NBtq!>;)v<N)d@+&+hJ<y>OPjS@030mTVc>vLG{>20D@i?=_&|@``%(BBip)64^wgx=CDg%!y&3T!R$iX8R*t+RT;)>THz2qL&TjbScH?h4pHh-b3Jp&KOH?_!RE%H|<mc?M7+vlq0n!F^wv5alqILlBtZFrTFd<xsi9&$1V*nfzC=H9;cr#>>YcYZUHb^EKfHk>g&f)q1=9a|&1}I6+)QyS4a_LKfs*`J*#pA#R5IC<AEiN(3`J(5x4yO6WCAygMaZ^Jr*`i|;*91F<9I`6j!pk$Am!{Decnj|@7dO&f>2CsXp&pzT)H@%@Md!WQn46*`Ddq@-D+Al!l=2{UjY%8b1XL4ql1>8Fu*&O1`e<8_vnpaZEX_<#<8lhqiJDr?&Qw^0yriJdg6X;^zK;6=@U%pO)!I}QN&qoe#M<OaaWA47P*jeer<g9Pz0ayd3Ao5OR54AUC8|ru+N~2asCz)u@Ek>O##%FX*hD##yb?NB@k*UJi5hn-(=c$F{o3DRu{Hq~n6EOT*$1#klXxnSFBBjnL9k`;0Tr&Hq$v$w*R`k<Cm$^oD(XIoC*@##swfRCs;TS#vA`?iZX}XVMU8P0b`0>Zs?eW>S*bm)<~oX*B}a5cShh{<mm5QUZ4;PICrBNH4J!c6x>#DOs(gEX01aJ~ZKzCKQgr?rsG8mhFC&FN!N8Li0a1LPS4oPJl%>N}jM4#g)!;G8Px1UH`#~ucHt8e*E74nRm8pw-^7=e5I72P2mP_AK)A59q=aEvHkm*Lmgc5OG3UX93nHb%TX66au!;+klQDNeCAq~n?)2_eZ0XeJEe8_7@g_Ab13+**k5`ZWvU<G+$PNjPl7o|q5A{7)?UjW(+^R<@Mm?(nHPB<3Yl~RJZn<2#0NfHwif;X$IV;-Q30Z-U4KSqq~?rpo3os?Vq)zS+9C$dtl7X}<ekjtpit8mI7r=Rr3#_M7xyEwqfq*#amq|)2xxNJC4SlSm{I4?EW<z3FCv`F!cU_r!zwOl;Ot_pU?gPb<4G@GP&RwMfykPw!QCMw_Pm0&4X$|P|eA;qqdo|3+}0lSG%W#<(rl-(+dC{PGPqTWxX7QOtQHT9jc^Nmw>LMK<LOm>brQh?7w@R9T6>_8+r5$vCyx1({TG`@rk@r;)f1zGek><mC1IXx{$sFC8r4uvLnmLsikReGo=mU%r&l0z|BgIR}}JfnoRi$uzSDr8LC%0WI|CxrCn6l3`M*vAJg$w!ot;VL>5LXe=wS`EUQZnT;ViO&o{G(Oxti_6ggA;JTKt>+#A`m47@B`qb=$=fD*FN@U%vx@&5WYJJZFS>5!YK0nMOUnr#yTrw*%@P5@T2U$QU?HfK66i_DcZ=oF5j^3$rt#GY>Hvyh8)Kz$O@BK#y;!kOZc1~mS?NEmqy`n*mB4`}K#;&|oZ~{qur+y@jv!c_Ze_3#+Yb>cUM5tMOk!%;fu0!=&pF*USL2H(>(CDHw9f<GA?1^0HpW}ieJbcqp^8<uV>eTgdq&s4$>%wY3-r`NJkjYUDa`iSg_8($Gu0cSZ)hhJDa+;}Fcjm_C3^#AenKVWb!Bmw#;=_TmI&v{$mXLX%)S-`!I31k4)fqA$;(Tf6rvHTRg<RyZb`Re97+`;%!uS{>|#0-5X}2HIYF!v6BT5V0_3AW>$)o!fVxpK(Iy5Om5i2aN2YP^@I|!k!v&W$S))v`bsut|m#DCNa57miV`|bO%^i+C@Jr#;3air-U{u*7@xFnJONae9d==nfG-_|RQv>=XBHs}<6Py5=6L~t+x4Ii6#?z!MI=#ZSmVoC8s)?DI6x5?l&WdOym?Vf6N82urXeK}K)}VMcr$|xYqis|QjK%H{=kuJd`KwGdMLMW-7BZZ!sfaw+uaxhV#z>amv%AL0?$Q+R{(z}mVqU9GqiyM*X`KZ-g#W9TX~l)WYR>I?lKifO=I}-;$|NN(4#LkPVqE}5VK^9eh{sldjz(Ll=~XIr)mq={be<A^BP$qe5wO5Y;|d*a4Pj<~u3c%o*Z~$@TvoA*?w@gLcCm)r51k2Hh}I~*fpc-71vSy0WM4j!3MoQ2NeG;1Mzg~f{3m>7c}U9e6rwQFEo1`ew5A{nF0c-x<BrL#gvS?1rn*kx7cp-%!ju_!x!Zs_CIewp;LYHO7&!|_v{#4NqTX&lOHMA7MRks3`dcP##Dm}1<gy%NhIfT6zpj=1<Gj$M6#SsHnXnC&{9wZT(N;##TV3v$fg}#B4E%&aDw7ZjBQ11R@l0c%h@#z*$(knfxK+yNh13ibs)F1R5R_1%eocOlB=6J1>{o6<ueXzDW|^vgbV888sXbSOi1UF`kPsLo^Rsh8i&F5W7=V*<$|uYX6WLJA++;Ph<UEU=_`zZl22}qN07ehhn=GN)NkLL+g>dY_Gx|}FoI=J;DIP>BilHH4u3x?Oz`CZ%6!4u`5xH$bm+B?Ezd~;URl`^oq@9I|n7NpkJ&w+8h}KX{m*dlLYj8Hf7&^$ZMqJ=4XvMo|4QMaO2X_ZV2f#LF+&R~McH`eN>xtxatum4C(7lU$lgc~JmW0-7C<Q|s&}a^FBL(j`l)k0@7GCL6fItw-kcOA4QhU6$jqmy^Xl*GGCwj#;4}z<D$&HCaxxwMT%jR6`(m;&C5@)1(09mXGXtc_QL>@u)iixgaqXtg6pOMFsw4CT+LUSx+;Zb|t)_2b}`=G3Q=^!PG=RdxN&-&aepC6)!rK)1r)093x(&5I}^>LxRv#yEmLn(sxvw{S@z9vv6h^Z?Yu?YKHDFiWOie!uFb-omJpc3i>F&?K~OxHRieG^dsO9kT-QqW>L`$j^5+SEj)B0#{VW}F9*GKBKw#;Y?1j1FgAAt4=asR_bLk^^k2M`+ia#GkPp(-IORx{S8Y*5j~&f)m*D+`~D^gRJAgM}p2zMq7ts2M+P9?}Mxol2klTQe=3<BQIzyuaK%yB9T|~1R1}XdajPDi68K<!m4cB!$st|<dGEiC@So>{kZ0<y%4Ar5u;~tM^S4w#Fr9?J5W_=p9aO|q7)G>ilWte2O}?KJN2`JX8r{NzF%dsEJc~1m}UZKGExD1ng_)c?YA2hL!m?}gZvEhtEev4%rR6l{ovW{tXADPy_hdD3x=ZFK~k$<cEh&Y5=B#Nu=cuTyTQ3}HNws`k8M@VV-vjjW}cGgZM{B!3D_$I5hMGPZz094N(ItsA{w-l>M9EDNG@7xU5muDF&3poV`xcv13qch)VO?3qP|tBPF48xs3=AsE9889a!$kfT9veDn81-dH;g#IebRl~ugU_$td^eygW#ZAgnwu(AH4{bu%4*CPT*NnxJM3RMhe5K=<qUAI+*H_QX8Mr00I_5Rqe^$>t51E#Lh|PohotZrs`5*t`Uc-HYpTF=*_ZG?HcO1Qa5o+nP<#Frjh&{|NqdTQ(xLv+}a_239UlvSu;|Py#d0q^$ba5(f!+|^$TxkJ5?p~K&ivd*7C|tZl397<+ETG;Pui4ZV@-BW_K#fv*%DP^tb|JWdnj&Nb8D~-)VfIbfvJeVhPn%9f+qZSr3IQGMYc6q&pS7_NJk!<*QoDy(>7@wT~E*O8GPoQ0;WO8e$AI(m@Ga9AXv|2{jEBso4R37ibWmnt1}8N6h5JSvH+2)ocX)&yhjC`MHzUdW6WeE%<s<8SurMq%8)&8C&J*aw-Ur7pIE|TxzPx1UR3bpq*lwy%EaC;Jz^PC{(*pw6gbG@LH{YqALVr6GJI$7{L{-df5UL0ahN9l!(}N3>(W<Zz`jGV6iC$-$eS14D_wCKp)WIXG9LhPy>(B2*{lw<sH>m&zaPfjBZ{6c2_O72FFB6y3kTD(I==RtWvaeC?AfqT%3$z7P2kS#%+=kcsQq7&WusTdUWM4DLY#&2p>Wcm1S-#+C^K|RxJmA5+V13DYB@CPZ)1cs|LfXxy5mkgccsv<|mUu674i`iXkya!{n+c0yq*yUHCFTwhS^RxXS^YicF1uQf>rASWud09hRX{HD&{5$ea(5`xJn{Wl{WWu!)rdX1eLpT|H04IpNx&;^hIMfR^ublPH-Nb;U%chC%z;xY4*Y<kho0=3W5daujUlTLZ(sD|5->4c}g&K1P`hNnx2R`*FmmbTE$U6+7qo*g1#-uq7akL|BN>z(%OP*FF7kNmGgiGVGJ^dYwPR5_ZTn4q&WskONvtBEQgnz-8UJcyBa$5D*<F1xHSZCfZ*qz*H=ZR#nb|^ROK%;v}nC1km)z<@BUvM}l-p^{iEt_N172S+YJ1aqnkE6*)$Zx|%__>yJ+xTGs;vOlVm}8Q`i-X?I^YO%?N;BA!ULSbSKENo$`7#YSRTH1`e;_~wTnve{7fK4LnRD%-UW5|DhKTgRURoRL)vcB`%(f-E`}T>|r}tpr>wYe&g2G@M*W7Zh{9qB@ysrl*|oX8AAiuv6eBV(5@cy)X@Uzp4_=)5nmKfGu@YWIhGC`WB7hD}vDmMHZLEiztXZ)L~D!VK-@A^~Migd-`d{%s0H|NoCup2o!Npl&m1I9`?zuK5Fs#_gnI4NMAnd(XFszvU)megnt$+N)<Mymk=<U&0O?w?EAZq$4f8=Jdm-qkG0=KYMP`6Qns0bOVipMk>?UsXN!K7@YQXclj-2lBaMg30ZLUu5&|Z=EZ5A}s{N)^qkF!{l!#+18Ko$j|ITFuf|^Ll+n(fvMXo0`&aIywFB760#1~}{Aj+;0^8ah2i-Ol73I^6L&G>WmJ&FqcY#CD}|J>PLx&T?bg9QMfFffKnA3A&6Um{+`P*dFcGND38BI)cbQNaQC%&gUSV@ELi8#<9fV+8%qqLh7`a9%4zMpet>x^=lt5_=a+bB(#Osb)~9)%{^e#B^fi!89?zX;3VSEE~Gsz<~KNquSK~o<}O3LK8`|H!7`U*Eu_C<T}mTCRaqibhw)FUl^3F)TxlF!YwlqKu+_~)msAY*kHy+8R-5dwD&IcM`&!86R`hTF_=}tttWxyILYLyqK=h^RR~le@mfUz+MYS42V7a))I+sGbVh@eiZS~!1|E>RMXGQ+83Kldq_S4@e^rN?T<Us}ZYk-)q4?`^-6T>a%64NTR?MInl^hd-%wpv^Yi1~l`2EDC8X#9o8VcMuHLo&kxDwU0;#dg=72nXr5NTj}2ON4anK3AUZ7<qN79G1ei9)a^lI4lb$dt)&!ALdISvq1+>rd<fVZD%2I5v``A3%H|d&g5?h&L6`$keYF=mj1y+1ITIHWEyVC?@=_g{{uw9@K7Nop4>n8Y1JYZoQ-;(Fpd>vYgrIv=QuIkTgP*7@a3OnSQz<9u2P~(KORGI5}-rr3#LE@vyAioOc_jgq)k2GElcF+QjmFAV$b@O9br(h@sg{Tawt7`HEdr>los+^c1D3u7J2dPqww+p%mSNCQyrba+nf1u+xMGo+qW``%4MBi7SSkay(cidZ;y$N~UfOI>d5jl@4lZHNuJQr)yA#GIgc3K#Xc#s1Gu7DoHy*F8FS5-Xw;X8yew#DB$8F<q&V9lC=p6h*GDSMMW032?c$!Y1JaY6dW|CkY=0`HILX7WdNqXVVh7uF>^r?$E0!MR!$P*cF3i|<@5`RA9?ebl;LpejEkZaR^a$Hu}4HAPeJ)h6zg0p9ktH>SI;fjByBN^GY-Eh@{j~BoK~)kJK*`!)sfzJlyJx5{ISZ!81<qQxn^ptxZ6)jrHk~)1wnDyI7i!3rFw5^!e&7aEUKv>2AG-llo`uKx&(+458%RK845)R?PhwL5DtULC*QYZZ<0SH3oESDWF0mu?7)tqXA<pAAk<UYt+xO;ti#cmf36o!O5ZtiuX4|A+|Q`)WM#WnN>aP#XAb0m&7icE$eYFU%VxctN)#p?q?#!FXOgtZN;WR77&TNLw`S)$B}`Wjn(-jiR!#929I9;f(?nu^0J}x=%dl^)VuzZggj+2auSlc@EmoGypDO^28>34>@aJT7j+Bbz342E-P6*;6L8UaFyQfRjB<iF4QLCDjt*Q9&+ayRRulWJt)blnmm9Fq=s9-gc;-OS|OZZ<rY0z9iH_zj=KV8Kd2r>xS8OtjGGl>H8n)1P@U1Q1k#yl9dcRwz$xpb8P(aG*BAo|*UCAD^c$smgIHHXj!>W{-{>F(u0Z$a~I9dIeQoG2Nyx)jt}4hG$2@ZEm4L9vo0xVL9&0n#gV7ii-wja+V=8Qb760AX7{-QT_2bgOW+r_H{6buwhfce1~c58y4C{@9w{2k|bwgD2T&hwlvB?ld^x&(oLXJGykX2a=t=)Q7;-?;bSwn)M|;2KH^Kf#|wd-iBkgG4(+xp8oug=l=&4PwH?')
).decode("utf-8"))
_ACTIONS = _cross_graft_copy.deepcopy(_CROSS_GRAFT_ROUTE)
_LEGACY_ACTIONS = _cross_graft_copy.deepcopy(_CROSS_GRAFT_ROUTE)
_REBALANCE_ACTIONS = _cross_graft_copy.deepcopy(_CROSS_GRAFT_ROUTE)
_trace_actor_action = _cross_graft_copy.deepcopy(_CROSS_GRAFT_ROUTE)
unit_actions = _cross_graft_copy.deepcopy(_CROSS_GRAFT_ROUTE)
_CROSS_GRAFT_MECHANISM = 'v27'
_CROSS_GRAFT_ROUTE_NAME = 'stable12'
_CROSS_GRAFT_ROUTE_SHA256 = '5ee61bbcd473c7bb59ec284b0992dc881eac24a1008463b6d0d4edf772d131ff'
