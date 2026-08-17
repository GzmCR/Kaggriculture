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
# Normalize route payload names for the shared V031 overlay.
_ACTIONS = _LEGACY_ACTIONS

"""V031 runtime overlay for a frozen complete route.

The builder concatenates this module after one decoded route source.  The
route source contributes only ``_ACTIONS`` and the official market constants;
this overlay owns the runtime market controller so adaptive/V27 market logic
is not applied twice.
"""

import copy as _v031_copy_module
from collections import Counter as _V031Counter


V031_PREMIUM = ("MILK", "WOOL", "STRAWBERRY", "MELON")
V031_CUTOFF = 648
V031_MAX_ORDERS = 10
V031_MAX_BATCH = 30
V031_ROUTE_LENGTH = 719
V031_CONTROLLER = "raw"
V031_ROUTE_NAME = "unknown"

_V031_STATE = {
    0: {"last_step": -1, "pending": None, "stats": {}},
    1: {"last_step": -1, "pending": None, "stats": {}},
}
V031_STATS = {}


def _v031_get(value, key, default=None):
    if isinstance(value, dict):
        return value.get(key, default)
    getter = getattr(value, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(value, key, default)


def _v031_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _v031_copy_action(action):
    action = _v031_copy_module.deepcopy(action or {})
    return {
        "farmer": list(action.get("farmer") or ["PASS"]),
        "hands": [list(x or ["PASS"]) for x in (action.get("hands") or [])],
        "market": [list(x) for x in (action.get("market") or []) if isinstance(x, list)],
    }


def _v031_seat(obs):
    return 1 if _v031_int(_v031_get(obs, "player", 0)) == 1 else 0


def _v031_farm(obs):
    farms = list(_v031_get(obs, "farms", []) or [])
    seat = _v031_seat(obs)
    return farms[seat] if seat < len(farms) else {}


def _v031_align_hands(action, obs):
    action = _v031_copy_action(action)
    expected = len(_v031_get(_v031_farm(obs), "hands", []) or [])
    hands = list(action.get("hands") or [])
    if len(hands) < expected:
        hands.extend([["PASS"] for _ in range(expected - len(hands))])
    action["hands"] = [list(x or ["PASS"]) for x in hands[:expected]]
    return action


def _v031_step(obs):
    return min(max(0, _v031_int(_v031_get(obs, "step", 0))), len(_ACTIONS) - 1)


def _v031_stat(name, amount=1):
    V031_STATS[name] = V031_STATS.get(name, 0) + amount


def _v031_reset_state(obs, step):
    state = _V031_STATE[_v031_seat(obs)]
    if step == 0 or step < int(state.get("last_step", -1)):
        state.clear()
        state.update({"last_step": step, "pending": None, "stats": {}})
    state["last_step"] = step
    return state


def _v031_tile_at(farm, position):
    try:
        x, y = int(position[0]), int(position[1])
        return (_v031_get(farm, "tiles", []) or [])[y][x]
    except (IndexError, KeyError, TypeError, ValueError):
        return "LOCKED"


def _v031_actor_trace(step, actor):
    trace = _ACTIONS[min(max(int(step), 0), len(_ACTIONS) - 1)] or {}
    if actor == "farmer":
        return list(trace.get("farmer") or ["PASS"])
    hands = trace.get("hands", []) or []
    return list(hands[actor] if actor < len(hands) else ["PASS"])


def _v031_weed_action(obs, action, step):
    """Common actor-local DIG/retry/catch-up layer for all three routes."""
    action = _v031_align_hands(action, obs)
    state = _V031_STATE[_v031_seat(obs)]
    active = state.setdefault("weed", {})
    farm = _v031_farm(obs)
    positions = [_v031_get(farm, "farmer"), *list(_v031_get(farm, "hands", []) or [])]
    unit_actions = [action.get("farmer", ["PASS"]), *list(action.get("hands") or [])]

    for actor, tx in list(active.items()):
        index = 0 if actor == "farmer" else int(actor) + 1
        if index >= len(unit_actions):
            active.pop(actor, None)
            continue
        age = step - int(tx.get("start", step))
        if age == 1:
            unit_actions[index] = list(tx.get("intended") or ["PASS"])
        elif 2 <= age <= 9:
            unit_actions[index] = _v031_actor_trace(step - 1, actor)
        else:
            active.pop(actor, None)

    for index, (position, intended) in enumerate(zip(positions, unit_actions)):
        actor = "farmer" if index == 0 else index - 1
        if actor in active or not isinstance(intended, list) or not intended:
            continue
        if str(intended[0]).upper() not in ("PLANT", "BUILD_PASTURE"):
            continue
        tile = _v031_tile_at(farm, position)
        if isinstance(tile, dict) and str(tile.get("kind", "")).upper() == "WEED":
            active[actor] = {"start": step, "intended": list(intended)}
            unit_actions[index] = ["DIG"]
            _v031_stat("weed_repairs")

    action["farmer"] = unit_actions[0] if unit_actions else ["PASS"]
    action["hands"] = unit_actions[1:]
    return _v031_align_hands(action, obs)


def _v031_route_action_only(obs):
    """Pure route action used by tests/benchmark for field-action diffs."""
    action = _v031_copy_action(_ACTIONS[_v031_step(obs)])
    return _v031_align_hands(action, obs)


def _v031_is_sell(order, item=None):
    if not isinstance(order, (list, tuple)) or len(order) < 3:
        return False
    if str(order[0]).upper() != "SELL":
        return False
    if item is None:
        return True
    return str(order[1]).upper() == str(item).upper()


def _v031_item_price(obs, item):
    market = _v031_get(obs, "market", {}) or {}
    prices = _v031_get(market, "prices", {}) or {}
    return float(_v031_get(prices, item, 0) or 0)


def _v031_market_inventory(obs, item):
    market = _v031_get(obs, "market", {}) or {}
    inventory = _v031_get(market, "inventory", {}) or {}
    return _v031_int(_v031_get(inventory, item, 10000), 10000)


def _v031_impact_score(obs, order):
    if not _v031_is_sell(order):
        return float("-inf")
    item = str(order[1]).upper()
    quantity = max(0, _v031_int(order[2]))
    current = _v031_item_price(obs, item)
    try:
        later = float(_market_price(item, _v031_market_inventory(obs, item) + quantity))
    except Exception:
        later = current
    return float(quantity) * max(0.0, current - later)


def _v031_reorder_existing(obs, action):
    action = _v031_copy_action(action)
    market = list(action.get("market") or [])
    sell_rows = [
        (_v031_impact_score(obs, order), -index, list(order))
        for index, order in enumerate(market)
        if _v031_is_sell(order)
    ]
    if len(sell_rows) < 2:
        return action
    sell_rows.sort(reverse=True)
    ranked = iter(row[2] for row in sell_rows)
    action["market"] = [next(ranked) if _v031_is_sell(order) else order for order in market]
    _v031_stat("reorder_calls")
    return action


def _v031_visible_inventory(obs, item):
    private = _v031_get(obs, "private", {}) or {}
    total = _v031_int(_v031_get(_v031_get(private, "shed", {}) or {}, item, 0))
    for inventory in list(_v031_get(private, "inventories", []) or []):
        total += _v031_int(_v031_get(inventory or {}, item, 0))
    return max(0, total)


def _v031_current_sell_quantity(action, item):
    return sum(
        max(0, _v031_int(order[2]))
        for order in action.get("market", []) or []
        if _v031_is_sell(order, item)
    )


def _v031_future_sells(step, horizon):
    target = int(step) + int(horizon)
    if target < 0 or target >= len(_ACTIONS):
        return {}
    result = {}
    for order in (_ACTIONS[target].get("market") or []):
        if _v031_is_sell(order) and str(order[1]).upper() in V031_PREMIUM:
            item = str(order[1]).upper()
            result[item] = result.get(item, 0) + max(0, _v031_int(order[2]))
    return result


def _v031_public_signature(farm):
    counts = _V031Counter()
    for row in _v031_get(farm, "tiles", []) or []:
        for tile in row if isinstance(row, list) else []:
            if not isinstance(tile, dict):
                continue
            for field in ("crop", "animal", "kind"):
                value = str(tile.get(field, "")).upper()
                if value:
                    counts[value] += 1
                    break
    return (
        len(_v031_get(farm, "hands", []) or []),
        len(_v031_get(farm, "unlocked_quadrants", []) or []),
        tuple(sorted(counts.items())),
    )


def _v031_clone_distance(obs):
    farms = list(_v031_get(obs, "farms", []) or [])
    if len(farms) < 2:
        return 10**9
    left, right = _v031_public_signature(farms[0]), _v031_public_signature(farms[1])
    left_counts, right_counts = dict(left[2]), dict(right[2])
    keys = set(left_counts) | set(right_counts)
    return (
        abs(left[0] - right[0])
        + 3 * abs(left[1] - right[1])
        + sum(abs(left_counts.get(key, 0) - right_counts.get(key, 0)) for key in keys)
    )


def _v031_append_or_merge(action, item, quantity):
    item = str(item).upper()
    quantity = max(0, _v031_int(quantity))
    if quantity <= 0:
        return False
    for order in action.get("market", []) or []:
        if _v031_is_sell(order, item):
            order[2] = max(0, _v031_int(order[2])) + quantity
            return True
    if len(action.get("market", []) or []) >= V031_MAX_ORDERS:
        return False
    action.setdefault("market", []).append(["SELL", item, quantity])
    return True


def _v031_reduce_sell(action, item, quantity):
    remaining = max(0, _v031_int(quantity))
    if remaining <= 0:
        return True
    for index, order in enumerate(list(action.get("market", []) or [])):
        if not _v031_is_sell(order, item):
            continue
        current = max(0, _v031_int(order[2]))
        reduction = min(current, remaining)
        current -= reduction
        remaining -= reduction
        if current <= 0:
            action["market"].pop(index)
        else:
            action["market"][index][2] = current
        if remaining <= 0:
            return True
    return False


def _v031_repay(obs, action, state, step):
    pending = state.get("pending")
    if not pending or int(pending.get("due_step", -1)) != int(step):
        return action, True
    trial = _v031_copy_action(action)
    if not _v031_reduce_sell(trial, pending["item"], pending["quantity"]):
        _v031_stat("repayment_failures")
        state["pending"] = None
        return action, False
    state["pending"] = None
    _v031_stat("repayment_successes")
    return trial, True


def _v031_preempt(obs, action, state, step):
    if V031_CONTROLLER not in ("preempt_h3_h2_h1", "combined"):
        return action, False
    if step < 120 or step >= V031_CUTOFF or state.get("pending"):
        return action, False
    distance = _v031_clone_distance(obs)
    thresholds = ((3, 3), (2, 4), (1, 6))
    prices = _v031_get(_v031_get(obs, "market", {}) or {}, "prices", {}) or {}

    for horizon, max_distance in thresholds:
        if distance > max_distance:
            continue
        future = _v031_future_sells(step, horizon)
        if not future:
            continue
        candidates = []
        for item, future_quantity in future.items():
            current_price = float(_v031_get(prices, item, 0) or 0)
            if current_price <= 1 or future_quantity <= 0:
                continue
            available = _v031_visible_inventory(obs, item)
            available -= _v031_current_sell_quantity(action, item)
            quantity = min(
                max(1, (future_quantity + 1) // 2),
                future_quantity,
                max(0, available),
                V031_MAX_BATCH,
            )
            if quantity <= 0:
                continue
            try:
                after_price = float(
                    _market_price(item, _v031_market_inventory(obs, item) + quantity)
                )
            except Exception:
                after_price = current_price
            candidates.append((current_price - after_price, current_price, item, quantity))
        if not candidates:
            continue
        _, _, item, quantity = max(candidates, key=lambda row: (row[0], row[1], row[2]))
        trial = _v031_copy_action(action)
        if not _v031_append_or_merge(trial, item, quantity):
            _v031_stat("preempt_blocked_market_full")
            continue
        state["pending"] = {
            "item": item,
            "quantity": quantity,
            "due_step": step + horizon,
            "horizon": horizon,
        }
        _v031_stat("preempt_calls")
        _v031_stat(f"preempt_h{horizon}")
        _v031_stat("preempt_units", quantity)
        return trial, True
    return action, False


def _v031_agent(obs, config=None):
    step = _v031_step(obs)
    state = _v031_reset_state(obs, step)
    action = _v031_copy_action(_ACTIONS[step])
    action = _v031_weed_action(obs, action, step)
    action, _ = _v031_repay(obs, action, state, step)

    if V031_CONTROLLER in ("order_only", "combined"):
        action = _v031_reorder_existing(obs, action)

    if V031_CONTROLLER in ("preempt_h3_h2_h1", "combined"):
        action, _ = _v031_preempt(obs, action, state, step)

    if len(action.get("market", []) or []) > V031_MAX_ORDERS:
        _v031_stat("market_overflow_guard")
        action["market"] = action["market"][:V031_MAX_ORDERS]
    return _v031_align_hands(action, obs)


def agent(obs, config=None):
    try:
        return _v031_agent(obs, config)
    except Exception:
        _v031_stat("runtime_errors")
        farm = _v031_farm(obs)
        return {
            "farmer": ["PASS"],
            "hands": [["PASS"] for _ in (_v031_get(farm, "hands", []) or [])],
            "market": [],
        }
# V031 generated candidate: route=v27, controller=order_only
V031_ROUTE_NAME = 'v27'
V031_CONTROLLER = 'order_only'

# V032 offline route/profile payload.
import base64 as _v032_base64
import json as _v032_json
import zlib as _v032_zlib
_ACTIONS = _v032_json.loads(_v032_zlib.decompress(_v032_base64.b85decode('c-qxnO>Y}nlKd|^*TMYIhxewoIn%<ZEkl-vm^Fw-1JjEI20e$Kxh?j;uWXS;R%K*lWWGnM_tb2eta{%sGcq#r%l};b*Dt^S_UrE#|MBJG<L#$U7rWu&zkm7X-~aXehv$#K{qp-ie*O3J=PwsOeE!?dA0Hpye}4CLF<fka+}u9@cd_0LUoL*Ud)!<EAHM$l_QTyzxA)J#e)q6_{;m1*ryn<)j~B!G@DHChoA=Lu{%Ldn@Zn;(oc(#<gv;Y=cK<s}`g8a0FP}f2yl(8-my7M@)6=(hjUV3K?pr$k?6f&AV>2C*Pn-MuV>{mBR<CfYmu_PZBRgsChlj_fAHTKk_-Q{~_NPvRXD{4$Tz$z8pYQJ9|NQ*FPoEFt6un{o6S+V4x9>K`@n1+UJNyXD_47Y{cp7J9`qFXM|MYcwq&MxqdAgWxpEi%)gOBSfdhq_6aF`B#c$wnKbLS(-25Gj#(Q}WTX_<T5m*aE6mMhJ;>CE_GGFtjP+b(>1_`Dx!Fi0))Y**;9D~IPdP4h{6&Cqn7=J8-1U)g(_mC<UMW@)hW>9!phe+$NF)|K2cIyCOMP~AV}M*QSxo(+A=>dB2vuhnyC@aDF@Xh_TL5Oui<*vQnMJ2hya73D3n!l0k~CofHX^@e$eBKyh1{r%?M)6aj}JU-ps-~H#e8McN%%A5;F11vJo4Dx`(m6U3zx3n8ZC)w=7QP^{W_RmZ$r1&?LvC41PE^qlh4LiqAxZAsT@=oUc)Z?6cxW+-g-svZM)5YBNx1P7}c5{()XGbF^J}#`c1>IfnAg8HUyM)^%9B0A7FxnLLBm8*j=gwl-G^HW_xn6C`I11rGdoXWmqX32w4??9KbJs@NF<}EwoXSF*jssg6o2k=L>JOMMXT-w>GKafxzO}f3P2L^uf6ED-Hf&q&uqjZvnEm<F)8p;-ht1>T-@am-uXjPEe3gNYrF^GFEtI;cU5v92-7d(XW}+j!w*TGOdP^Q$?~^r}vkexzl8ZC|2)Ob#e(#doCv5fcEW|_9?Ee6GB77$B!{}|@bWueeFxYW+guvt>ycjrr4a)=UpxCRVEp<Z<RfF%i9%2i{8WZEeItv=44v6DJX4qNL4xJ%~HJq>)Xs2K9JV+`zYGezlQQ7K!)fz#oJ+OR(5x;tFV5Qwa=`9ssBszW*Bgewa=1izPaOPBgU2wOLf0g4W$YZtx?S+84xZ)sKR(U#jj2=b;2i@Ey%{6d;I`Uoe$CxSIdp&J$Vl0k7!(Oe8zHjPRi+|@fI>F<8mzp)%b_NrdpGxAKbs7%Om28a{3QucrS2C|TlilH<2WvtL>%ix#y9#rAWSOy63jR8>9b$*f_wiy|t=(PnO_F}-tmBOJGCFF^HEVQ)Q$?t0rD3!yNFB!5m6HfP@c>#I;CShcx5>+g+8Uoft_O7Cc>bN@=(+31jTWB{V4-CZ5LAx*2Hti31}h){tc~(Rh(U3j3b~`p%vG&UO$W0efrxxp7-E%&Ar#Xayym-H-u0O}HMz4OOd?({104+CZ2%rDnHl2fQ_B6~dRM+06Zb7cl@d&bXo#_kS&8df8>M7a07$<OdJr)scP-O^qm1hJGF}pPhkXp5;qXE4A0IwC6HIKW<7RnyxZkIzfJ1uBPkFv{-kW3D_Q8Sz=+)M(TZ(SoGPiqmW0HGy<8#Ipqd%E=3&+PAm}+*r&mK~9CqVj1Vx6(eO_}z@u~`V&R;-birT=_Qh)HUx{I#WfyDp1}!nen+fO9z{f+XH{G_70{Z00Q{gRJx}w}g3YNUF?0P*9TOr?bK-=)IB8*2ta^4Wxlt1gX(k>1HWAl7Lg-^u-?TNnFV;Bb)6)#$>=$07xT*S9EeU&2I>Fo@k5^<(y~sCm0jDh?4Wd0z)8(B`=Q_f~O5B@xW(DJkc+3I}>1;8v6z=&EiG@lpe4ct@UON+D6`p3ls{*2ea|jWk3vP7O@Nu<MLLqN7&4g!}AYA>w+7orON=rWst~7#mk?*7R68cjketepb6s|d4Kn3+>};Zhh#>(Lp-#kDphWQrne~{Og?LQT2rr0Br>GI*6V=By{h>)In*{4DLu02pnk<170B7&zH1|RD%3}CJ+fu>RHDZJ>A5?On1GEgcO8-_BZnm8c_>Lm*xRs1jJ^^s6SJ*56s}E^Ci^9iO)v-J0_mJU3vI>mt=_HC9Gtgz;oLlFgu+jPcevM?ge--s2`o_8d+bV{aO|kr%2n16FtCBqczoYD>c-2AmkC4PtsMhnq`9k>y%3r%(Lq^Be6ZV@;uc1(r<mR5GVeSz;5ZC$-C!zsr_;!{6ere11J}i#IS9d#CSiUbU+pU5C-zkUTWR4#u#lA9PrX)|mq}6MWiV-!ee8_(L<HTk7Wv{$YuKTMlFFQ(Y>px~>zCM6;8uM#HIdx#k@Z}QAdt8?HH}L)vK&1}ueLMZGglMb_o11r9^~xYyxtXTJk-V#XG3B|Kzh8E0M|bc)eeER*>!BcZB>j5GFk_voJBu1HUss?uNDz~YBIF(dB9769X6nyP)D$t%@52ju*zBj7s$jz%4d-a7jJQ3jHBrd-9~qCk|HAjgSUT@GmH0!=}&j}e-UgNbcrDD6ZBS^JP~W9oIaX4EahaGR8G!uc}Hhi3$a)+<8+f;Nj{EtQeW*tGTj~e$0$HIQ<~ek4RfWVUe}S%3FhuCu1YkfsX+C0Cv<=51l@H%zy=*buz-E3iO4;Z3)0>YG?tQ8nv%T$&}EDO5Ch)|&XD-E{&2IGq%m+-2gk`9yJ9U9i;YzMAlqLS>=C<M^PQ<r3oLpIhMkBpYxBW^S-KYn<gxAgN2`Z>Z`8oo{kV16k3)+~unC|-Ig!6taow_*4p3gH?GccCg*r<oZZ;AXiP*uzPR+8`CWmrw5>DfMl3_H1XjG$7wLKWoRN#m>=_{K!L1Ax*yx^ry)c%HWMxAOQg1C@XGmkiUeq?L*6X(w~Y(|9bV@{evbzpvWjQ$V7*Qs}k`YVVg`^(xLnmiuK0;c0Xj%yz{`m{^)b$;b9<x8>yu6izLhUDZHG&LWyLseR?_z2=_kQz5&kIqg&dW>11K{=bjBD80XrEJ2E1OS!h^CxxM)Ro7s7kd)A=P<QN3n4Fb;e&NE9s<CfkD5#{0Z^|)iC2yXioW33XxVl__|wNP*fX01_q-#?2am#~M2u)Q$$lBI`RN`Izchll-$7QmNdcXdg;ppYGIaWICgQEKF3*96`N?ep9A*uytPsFD!Gc7Mu3|)u*cNIEL$_L6QiH8R!^q#f&?R?sglaI=8RGTbU>t-{YN2~SPpG@#b8rd4_^a^_Dfp5Y!)ERrA!AXQQMd>Sv1~WlF1;*yl-9y}Rdzw>$rHquYI@|@(@7QrhslWWB!@Xs6lxjUHVh-Mn}*`t8N}N*<iwQUr&uwgm5ysps%DWGq}X*zZ4_`;meMTQXli!^#M&#w3Ch`)&<Dm&)u>fTjb4WDl8^-}5n17vXewjb59W-iDXTJPnU+=Itt&{C^mQ+~v14?4?{^r80$Zg+?&JR>_mVa|C4h%H*C4`$^zB6Na{Suq<-T<PWud#DH@e^lRP6&?Y6^f-Qf5nK9XV6eBVra?(!>jAcgmSPX4Gk!#HkWITCxK1>?IryYuJuI5`!FY#Y?F`V2Y_*oU!#}Aw}%rK=LNxa3$#2M4{LY5uhh)NuM+#--8)JMG+)L4lenH6O!Tp8x{Tt2aj&}%$~LD50#?uy;zWbL0K<N>kjV$;A_ADGu$-H`F<gDUo}^_0^3BGY&4)=@}*-T1WE<gx#KVtIQ`gcWR5`|7KB@bv3*YH<foT}?$Du<X$`tU)=UwY9!dEp#OG&R6LuWt(hcl7GP*AS1=SNh#-~kZv$6*)hNjiLvFsFAq+Hk9F^$W#O(F40k;-^}tuAh=6<u;HG#S$ULAU-xyRGMmfwm+DC*qwzkW`Dq=k#FIPv?(%$^xZ0Z=`ovDV2tt;!q1DMiwB?LFJ6ZT8nI|?6a*)J?~syf{6OCw7yj7(q+HG1F^qGFk>;OrHv^G*q|-j%t5Wl$OO5^Bzm4f@Wqr_8h>$vQdMglJTO6^%)Qk<hH$1U@8pTWx+qa(!UFuOI`2qN)@Lxy{?c<OVB~w_Ek#MMx+!fM;=)-y=!OLD6F0@rhq6g6eTBSpxyTVgIm#7kkbnxKXZ$an!+gps3(`!47it|z`xnN7{kBfmyK{&1TDqvv!^?%MmYKL+mn3OFZr4=t!+$I=RfFWOLUjt^sPjrOa!%1}Rd%qA)QES1KLFgjx9x?#BguTdP&hO1>h;3JI79MZQ#<99zy>0IgBrF@*uQKGuy7I$5B7^~*`*#c)<|A|#Z;4^N)HR4K=O8gGVJI?h(W98&Jy-$0%pOy=KDu6VcUlQwE<YGOyWYPSEOrpgn|y#Eh(`1nGj02$sp(Ubg?&Fe`)Uw>)ST+d`WQ&CRK(?#F#Uz>Fq4g<tn*0%+D(JyJ!uDXkmjIDl_h8ZJY>V7XpA?l(E359N%C?WGPXVBy4{3jusg=9&lp|a$!BRN?w(Q$uc1$PfslvF;NV=nKYN9Fct2mwUi6oO;(j+%GIPql0nKD<kZX_%2EY?l}61R|FluuDqrng<%m+<VyV~#)`?s1%CfGCoGh0|t5|>1{aH#2YcO@h^`9ve0Ii~6?kiiu&L<YTqJ+B9BI&U?h=A!3xV5QV6O|Lk5(hoaCz8#S6E$I+(1^H7$b1=mg9%=dAQufJ=7LKMi$sbnYRMX*nq0N^JEN9Hsx+y}Zow+110@Du&x6zP!}xS$MjM+KK#5~y3xNzlag}54HZ|O+bDPVe+=R2CB0w!=c2a7MF`<)c`|nNciY5vW<B@2rp;Jicq6&!L<d8}NuQDVFZ?3#YxF~vrzye*qg1&Ml8LvS`xeyC)2^scLl7U|0%Kev6o*Dk?5^w&gLg65ljFRer`)P{$p`~c<^obh^;Y`BN6a47!pySLuz-)ArsvL1?7mm)z(a(gIUZ(nr6INokt>$bobL*5k44~w=l!z#Ok?`}SI;+6(fd(PVhE(ZcE!ps_c(ay`iQ-z=RS;e8Rtk5hp2{@X*`8%mb)|AR2*g0K937YQ_A3jR%YM>Q37Ul!9*3ZOxhNCQkEmAM0^*zaSmAy&r7sJMMk=gBvcN!$5Ms#Q@s5{foMp<SlbQn2Jl47qj#X`$05lg5A-Gob_%-Fxh$IAPklySv{b8Fm2z2AJO7QGabX=rdCng6J?BPJD7;gfvx@QBNY>TtV$*xhBHdUrm!rkX6_-*@(oC3^0V?@J-{4|a(AX<v~lGkEa89Cgyi}h&)*DE|+s!?l+&}~E@ElfS}X<jl%h1ErBi^Y_Mtk6{VblLCB15Xy<nt(r;LC~Ke-jX#dQ^kv!jme0=>nwzX=IhcS3bJPdhatoHDtcQWL6z{qwzZhBqy{WVeGu52<+9Ds;RH0QEH^#Lmr9j1W$J(_LSP0d4OjObTbCX$WJG<;Oa)_k&ZU5uuv$ei=~9-da%hNDq&O~SBbcB`sFQde6TJ1dQDUYv$-q_BUDW-dwJgx=VGJZ@lTJbpYrz2nuMDRS+!U~1s`0E0VX!B+8-W_aMlb=9xvFtP2IUL}D!~JZ2whC$=){|=ImEIOCGfKJoi{z71pab(H!Ij18?OR*4{R>RFSPE#bX=ABvi{gf%+xA1$q+n<Lv+a=4>@mMGVnkF0@A4KU8Sn?<n+3EP4;;wmcc8l0kgWzh6qqmu!ZE=7M{_z6P;U<?vanya;rKB`+%WiCn?vW?9<qqrvf@6YXyg3dY39OT7q7L66VBua{(|_M;r6NnJ>$$e4bHLn}Vug_HJJ>EuGmP@wb6h8y5#Y$7W<;g>P<zF|CT6*?L!Ex`Z1JFZMW&RgfQCWEgCm@=TK<8cVt`YV=l;G~R|>ZLapzOJ`%t7zQa>;|v$1!d{US%NQe(O@4z^f|LS6MXgnu6Az+J!Q*uen+OpBjFkoEC#eHk8B7QRNV<_;XqPxzH@jNNTL)D1Mg7x5SAdYUXed+5wg;FkZjT5uCLXWMfQRccLWWcUie?{31(Ry=52)&G3+Ohh5KmU{+t*hWOI2Iyop;b4SN1i%N@)e`VDicMPOg9+A?{V*RzWyYnd|9jcSRI>o90(a&Fq>N5as+bC7<9PVKSq+gi|R>t}9rfkw4U!q|Q9g9<KWrFtriq0_f=sB;6SKe1`#0<|C<YMw17kx~twmc-=dfDjJW#lXQHsL~1P|HFS}t!>30neKo8=)g+t+Dj~)UIwI375Wevl<!I>6s%pI5rK*P7rt6hTwA?DUHf|^3y{J~2mN!GQkVt+IpA?e^36oeG%e=DNJWv3UGRa7t880`^Iv|xx5S+#2?9%G!WrV1O8WETm2&*trw9|<JYZ(?XjahLK<0YY6Ei{oLpv}~btRS3M<S+pn*BhL|9Zx5LAhz%Z!Mj^5(%o@KsAO1abGr;j6$h>_dxnZdcA%4y01oO3MJRI23fq+F0=h*A`Uy`o<`7ua<OM2<&&8`<rSR5-Ci2Dz`CX-f7(i?eK$=qBgKB0%qlU%$>W`;x&jF>&qA*1*R4wZX<|!+2dDr50KB^fB#ZvjjmDg35#DE+gnC@2-r;g-z$pJ&&*_sjLocD<gvoMMoc&_XUm&5__TyE|kK7iV^TZQ`-7)EKjjApMi30IITRJ*C5#2M9d7h}??sgOi1q@rpp``K}g9GkNTC#wibBeqMc<_q}KkvD4w4XglS744^Wu}swR7DaZn>ToR9K09I(qEHDKNw^`s8w^q;NhM}{B(t;%Z`Wy69;~7r2wRCRJ3Cl}9@t{>tP252fbs1`@k+<7Z_6YhC}C6S&~YBab;en^PLe^m+9I~Kwp=8ei3&4j^WVJw%v-|gR34Pd^8$k)V=S#zT8bj|4}Oi=H`AtJ%PO_v1iB3n&?J`~WRBPnKp^nKeXopEG!sOHk!w|enBBgPrEN2Vw)rSta48aH5fKXpDkbxPj*=tA>Z@8#mL(bK@n4?&nJL8i_7qFnUznG>M8~+AitT@xm^GPNKevu3z}Qn5D5^l*XxR+TEpEVot~>FhYNbaRU$G9j#X!B7`t|N%dnj9i1kTS_OJu1Q*>Ym5im3vd&!MJXM6EvD8(_9a=`H-5F>|X`$Ra9ILWTQC^GF<buAtmm=mPcw9-dr+6@TFyN8<aHH4#KKW8S^24i9%FH(wGtQT#!*N*_`lXYk^UB-hHo{Z6&J1HYQ}?4)Yhp2Wk9aF^h7rmYzife7NZ+{UT$-=t*DAn%vZA}t3LFQw=_u?GGRpYQJ9|NLy9JbiwQwf}gU4(R|&ESC=wU;{t@_}yk=!K7q<tV;k|dKmOl&er}7=qMrg91MmAcqAEhg%u1c5abHgEh3|UXLO=}<seI!n!W_-4t(B7b6YIcBY{06(-}gAQTPFX9*0#vwN(t~IwPa`#yWk1`W~Eqp?s8jpvOh7k~JRprGnPwpfg@o<7A3yk`X<rt3*K%)8A^QGEpTa^2Bl4F-OF<s!|~+iFClGyy|7Cc3S-#J&S_FPlW45!htiru>tPtbu&J*dh`8Cy`Ga7b&di(tNxjFDJ-B!Vd8K~Oh@;jmtu|&ks%S`l#15Gq66=U6}^d=L<#K_nMS17BGqMTefEBMF5T?(%sU$94t?-8#f~9lBiQ}K4PU0CBY_~bt%L;SOoT0{Pe4+Dsv8E4kghze6su?u`H{2WyrhprO}=FL%@VMft#II(>$!#~HOzu?AQi1gTG)c*VFe8!<!ExlA}u97%Zkk-D{p1aHzlJ@m3(5!Z3HM_ILx;rC=z{2V9~TFPqJ|mB?VtoL<?DT5_4ODT?hPLAZ}Hqlu@c&sa%8%WWwZz%&aq}szPdD6Hha=?Z|QxwN{Wr$|PheD@oBoW$h@GKCxoMd4bsVYdWvZ<whzAN{-}HFUR(>>e#}kU8yBlWA^e{E157uiu^@$l(Sk#$ifLKo_HZ`%A@8(@<%C@qLzV+xMEGURnVa6*BEuR(br7yFi}zGa#snTL&QTdZB-kRF<KMe-khg8G=$=@w?f5dn6vK$1l7>>nUi6n6fqp1PfF3)<tm1jCF)_H0-8>3y~u$`R&kAOFc3Z`F7}piT1EblsLZh4bT2^ZIm0^RnkJWsa92gqH4NColfy}9p)drOCK5GoGD-`UO(gP382g>6q9wP#5cYHym|QkAz^$kmMKWj-b;!`J(M626AjzVV2uzRB=xmtipx#sn9ZdI+TPTPRVINBlfb!9n0+1PcFQxoYN2Pc+_`8WhfG2s3PHEa;4FwBNNkuohqqRu{xi|vS7}Q&|a>91Gx>cic=7O{9M{=matP}z%Xg0Gi<_YFiB%c^FActKh7un8$f1H0S2Yf1`&K4g6%^(`fwyA2-^SGeb*+h&LD2<fxZOhVWD!SyOW2<5jCYsi$<f{~GE@k*1q8LmI8W&+5ilR-_rS3U1TBW6wHc5X_4co0z$OXA#qCQTMb<b>@=)#qibluICv0QnXC2p5;9yRyLMdr<Fr<0Po{;hi+ux4h(BIQC$P!?j&6(d?=#AR;9pQ5b^>?|gj0nc8=1!5t!H?afeX}ZLcg)&IPG$+Mu(#s4%i@q5ydNr!hiL8Y_+0`6sil%p+%HQ&_&@hJ!n$|w4Ke-84O5v9XF*znv@Pcg|gY(r}iwX!^A*nyD4Mgj4P0ot7qlLZBC`o%tyfvyE+R<~BCPfW2Q7_o#q-EqEi6T{Ng|<M-mx<|x5G5m!*-^A&3wEFmJ8hga`$5psmKyjrBMdG|Xpm4p-~@Tew=rh*rdPEnrNJ}-o7ob}96hH(W|<;+9ji8<Sx&bxuVn2(xeD866%kgNkcDK?WRXlm>79Dxq+otPl@VL#ny_ae#IS4=*az8qO%~f)B+@w$jYnntQ>2O+U(E?o37(5cez4a-ugXY1&pzNtZ}s>}A>wlBsHJ!Cpfz23F+wg%oXSZ0HR!h-eM;wd2{bIiQ=*l;6Vh|pgcMq6Wg|wJGb|clIFSt$LL5p|KQ;g}(nt#lcMJ-o`^A+4EdLBN4MKO9YguXTFY|rcI2atY4UtPe=u8Y;7IRp|Ck%z5qNIGLK17L*#Sk_pkAkx<n-{0o2)nBDKeiOT17~7M*fzaG!{h;Y<^&2}jABoYNUqDATTl&04P&;jbDa|6NLHg$b#8DcOf3CGa09SI^g^esiWpO1rFdg3Dva`>6j_qI;V4d#poCnjmz2`PNJEJXI4cW&gm@syhdNt^R67pSP2Xb`GCe5MQM;Yq#&I#3)RC-F#GV_rd%f$d7orXH$o<?DM5WFvA0o0{*=?kEX0>}g#Sv+uNX}4^YN9dLX?}JjYT1ZN5u=KSj>AAPg`f&5aI{?(t8`@Pedn6+QM05CvB2+nrA@r#6BV&ypwtUrNhw34Q*0uMldUj^3Jz6-nmCH3X=VhNX{mu>4H#a2j~;$#JD0!=IxqHGnyCXtCycis8c9m@@Xp;o+~eFT+|vXdh>J6V&TuhOW52L^OBFabsxRXR_@*kxCC;Hl-fu1GgF5gLL5LQXq)1sIv`IkCASHrvD62{;O8deY=b2kfI)9qfQfsFM!TRw$FzTcTDN0)?l|P2bB7h$y^+nK);nFQhpN_dbnYLurRw}|WhibKhANG7Lb<`wk0-Ar_En7wsJPa5w?)O$cu%ah;sM#^3DqB(|B>+Mu8(xr}6uC5|KorezZXma)nk)ewJCNdt+X-wORUVrgWh@V<xF*HA$GCQP5sqoaFPN_6R<9VN%tsH3d!)re9Ssrxk#sCc9tP*1sjMI3V&;OfY7uT)TpH|_mQ{z%TO1~mVaO_u*<)Q&X{k&mO^_i+Af%ih&Ql#}VjLwPs0nWh;1#cuMoE7v;+eaYwyFj@jSj#~pE>|+7;mcx_W}gKCb=t$C?V^d)g%dwiEt-J39_IPDTmZ;4N;0h+rbY&eioqzDGIhIP)b<QHjjCmO1%r_d2ho=#UhiG5@xXxT*X#v7iDmaSitne6+$hxF(lc+N=1-K5R{Mo(NtPPP-Uq#CI@NPKyz4M^>O682*kFL9jc}Ujl2d$&0G(C;r_UZO&s`W*y>h%xE?kd^Hvg0IOSRv>z8i@t%j_==lsYMh-_htqxx&OGA*-<r~y|ir_L$uiR>MXeb_qXEHxQfs{TLbB8q`ZL}V~aIX}6vNfkdsyQ1U@QaKSZ$q|L9q=T*ivFL$9BHO2^0|o7vJB!T(ALMGz^Tu&j3_rbdxQSdnb0d?!GaEI!%}B|Ufgdjbiz-lQFEeTRl-<D-@Wt3|F)`l{T6;4~bOqupcR|m3G%_vMh@QScC{vMIIHV8^T0t=DQZQ;cjda4=s-zA_bIg*0o`D#OUp*(~e>F15UHzdv@m-;NlMhPu3=%TmZ-=YI@+!PqY*lbau?}3WI)|4-Z!uR=IW!uaoyOas!k_FCvh8rSo^sYKM|?>`gwn{_fFQN_R1-{5Q%SMb0u7ps;;(Y`O~ibb?tUo-{pmpQnGdQrPjT#<T_%^hNsLb|Gg)b3#4t;c(rGlc%4y|0fbuQbvaYjB6lp~<)SPHk4lh8j&?~Sa13T5UnE4TJgZMfq0+B37q_qHOa^#rsTAY3wBP}V^1upvt)r&@s{)YZIwcyZtXA1o6TRMmWAOdPKP%5zvyk#OOmSrk7PSvyx(Wse|nb)_a4F+*qB0&efJ^vlQHjNO>QW;;;kfRT8{t2*b=KNzCLt6^PulFybz8wiQ5()d0cvI-ytuo$Sp&BEptyvL0`WnmK{U3Z*5AX')).decode("utf-8"))
_LEGACY_ACTIONS = _ACTIONS
_REBALANCE_ACTIONS = _ACTIONS
V032_PROFILES = _v032_json.loads(_v032_zlib.decompress(_v032_base64.b85decode('c-o7O1pomR0J#')).decode("utf-8"))
V032_DISABLE_TIMING = True

"""Runtime overlay source for V032.

This file is appended to a frozen V27 route by the offline builder.  It is
deliberately self-contained at runtime: replay files, notebooks, identities,
scores, seeds and network access are not needed or consulted.

The overlay is conservative.  It first applies a one-event advance/delay
proposal, then runs the V27 price-impact reorder on the resulting quantities.
If route confidence, market evidence, inventory, storage, repayment or the
short market rollout is uncertain, it returns the V27 order-only action.
"""

V032_PREMIUM = ("MILK", "STRAWBERRY", "WOOL", "MELON")
V032_MAX_ORDERS = 10
V032_CUTOFF = 648
V032_TERMINAL_CUTOFF = 672
V032_MAX_TRANSFER = 30
V032_MIN_GAIN = 10.0
V032_ROUTE_CHECKPOINTS = (96, 144, 192, 240, 288)

V032_STATS = {}
V032_STATE = {
    0: {"last_step": -1, "pending": None, "route_hits": 0, "market_hits": 0,
        "known": False, "profile": None, "market_prev": None, "own_sell_prev": {},
        "last_checkpoint": -1, "market_observations": 0},
    1: {"last_step": -1, "pending": None, "route_hits": 0, "market_hits": 0,
        "known": False, "profile": None, "market_prev": None, "own_sell_prev": {},
        "last_checkpoint": -1, "market_observations": 0},
}


def _v032_stat(name, amount=1):
    V032_STATS[name] = V032_STATS.get(name, 0) + amount


def _v032_seat(obs):
    return 1 if int(_v031_get(obs, "player", 0) or 0) == 1 else 0


def _v032_reset(obs, step):
    state = V032_STATE[_v032_seat(obs)]
    if step == 0 or step < int(state.get("last_step", -1)):
        state.clear()
        state.update({"last_step": step, "pending": None, "route_hits": 0,
                      "market_hits": 0, "known": False, "profile": None,
                      "market_prev": None, "own_sell_prev": {},
                      "last_checkpoint": -1, "market_observations": 0})
    state["last_step"] = step
    return state


def _v032_copy(action):
    return _v031_copy_action(action)


def _v032_is_sell(order, item=None):
    if not isinstance(order, (list, tuple)) or len(order) < 3:
        return False
    if str(order[0]).upper() != "SELL":
        return False
    return item is None or str(order[1]).upper() == str(item).upper()


def _v032_qty(action, item):
    return sum(max(0, _v031_int(order[2])) for order in action.get("market", [])
               if _v032_is_sell(order, item))


def _v032_visible_inventory(obs, item):
    private = _v031_get(obs, "private", {}) or {}
    total = _v031_int(_v031_get(_v031_get(private, "shed", {}) or {}, item, 0))
    for inventory in _v031_get(private, "inventories", []) or []:
        if isinstance(inventory, dict):
            total += _v031_int(inventory.get(item, 0))
    return max(0, total)


def _v032_used_storage(obs):
    private = _v031_get(obs, "private", {}) or {}
    used = sum(max(0, _v031_int(value)) for value in
               (_v031_get(private, "shed", {}) or {}).values())
    # Seeds are a separate slot.  inventories are carried items and therefore
    # count toward the shed capacity once dropped; retain a reserve for them.
    for inventory in _v031_get(private, "inventories", []) or []:
        if isinstance(inventory, dict):
            used += sum(max(0, _v031_int(value)) for value in inventory.values())
    return max(0, used)


def _v032_add_or_merge(action, item, quantity):
    quantity = max(0, _v031_int(quantity))
    if quantity <= 0:
        return False
    for order in action.get("market", []) or []:
        if _v032_is_sell(order, item):
            order[2] = max(0, _v031_int(order[2])) + quantity
            return True
    if len(action.get("market", []) or []) >= V032_MAX_ORDERS:
        return False
    action.setdefault("market", []).append(["SELL", str(item).upper(), quantity])
    return True


def _v032_reduce(action, item, quantity):
    remaining = max(0, _v031_int(quantity))
    if remaining <= 0:
        return True
    for order in action.get("market", []) or []:
        if not _v032_is_sell(order, item):
            continue
        current = max(0, _v031_int(order[2]))
        take = min(current, remaining)
        order[2] = current - take
        remaining -= take
        if remaining <= 0:
            return True
    return False


def _v032_signature(farm):
    tiles = _v031_get(farm, "tiles", []) or []
    counts = {"COW": 0, "SHEEP": 0, "GOOSE": 0, "PASTURE": 0,
              "COOP": 0, "WHEAT": 0, "STRAWBERRY": 0, "MELON": 0,
              "TOMATO": 0, "CARROT": 0, "WEED": 0}
    for row in tiles:
        if not isinstance(row, list):
            continue
        for tile in row:
            if not isinstance(tile, dict):
                continue
            kind = str(tile.get("kind", "")).upper()
            if kind == "PLANT":
                key = str(tile.get("crop", "")).upper()
            elif kind in ("COOP", "PASTURE"):
                key = str(tile.get("animal", "")).upper() or kind
                counts[kind] += 1
            else:
                key = kind
            if key in counts:
                counts[key] += 1
    unlocked = _v031_get(farm, "unlocked_quadrants", []) or []
    hands = len(_v031_get(farm, "hands", []) or [])
    return {
        "hands": hands,
        "unlocked": sorted(str(x) for x in unlocked),
        "counts": counts,
    }


def _v032_signature_distance(left, right):
    if not isinstance(left, dict) or not isinstance(right, dict):
        return 10**6
    distance = abs(int(left.get("hands", 0)) - int(right.get("hands", 0))) * 2
    distance += 2 * len(set(left.get("unlocked", [])) ^ set(right.get("unlocked", [])))
    lc, rc = left.get("counts", {}), right.get("counts", {})
    for key in set(lc) | set(rc):
        distance += abs(int(lc.get(key, 0)) - int(rc.get(key, 0)))
    return distance


def _v032_profiles():
    value = globals().get("V032_PROFILES", [])
    return value if isinstance(value, list) else []


def _v032_profile_key(profile):
    """Group duplicate route samples before applying confidence margins."""
    key = profile.get("route_key") if isinstance(profile, dict) else None
    if key:
        return str(key)
    checkpoints = profile.get("checkpoints", {}) if isinstance(profile, dict) else {}
    try:
        return repr(sorted((str(k), checkpoints[k]) for k in checkpoints))
    except Exception:
        return repr(checkpoints)


def _v032_market_observation(obs, state, step):
    market = _v031_get(obs, "market", {}) or {}
    inventory = _v031_get(market, "inventory", {}) or {}
    current = {item: _v031_int(inventory.get(item, 10000), 10000)
               for item in V032_PREMIUM}
    previous = state.get("market_prev")
    own_sell = {}
    if previous is not None:
        for item in V032_PREMIUM:
            delta = current[item] - int(previous.get(item, current[item]))
            own = int(state.get("own_sell_prev", {}).get(item, 0))
            # Positive residual inventory growth is the observable proxy for
            # opponent supply.  Town consumption is intentionally not guessed
            # here; profile matching uses a tolerant band.
            own_sell[item] = max(0, delta - own)
    state["market_prev"] = current
    return own_sell


def _v032_match_profile(obs, state, step):
    farms = list(_v031_get(obs, "farms", []) or [])
    opponent = 1 - _v032_seat(obs)
    if opponent >= len(farms):
        return None
    checkpoint = max((x for x in V032_ROUTE_CHECKPOINTS if x <= step), default=-1)
    if checkpoint < 0 or checkpoint == state.get("last_checkpoint"):
        return state.get("profile")
    state["last_checkpoint"] = checkpoint
    current = _v032_signature(farms[opponent])
    # Several training replays can be the same public route with different
    # market tapes.  Confidence must compare distinct route families, not
    # duplicate samples from one family.
    grouped = {}
    for profile in _v032_profiles():
        checkpoints = profile.get("checkpoints", {}) if isinstance(profile, dict) else {}
        expected = checkpoints.get(str(checkpoint), checkpoints.get(checkpoint))
        if expected is None:
            continue
        distance = _v032_signature_distance(current, expected)
        # For the confidence margin, profiles that predict the same public
        # signature at this checkpoint are one hypothesis, even if they
        # diverge later or have different offline market tapes.
        key = repr(expected)
        previous = grouped.get(key)
        if previous is None or distance < previous[0]:
            grouped[key] = (distance, profile)
    rows = list(grouped.values())
    if not rows:
        return state.get("profile")
    rows.sort(key=lambda row: row[0])
    best_distance, best = rows[0]
    second_distance = rows[1][0] if len(rows) > 1 else best_distance + 4
    # A deterministic profile from the same route should be close; allowing a
    # modest distance handles harmless WEED and timing differences.
    route_hit = best_distance <= int(best.get("route_distance", 8))
    best_confidence = max(0.0, 1.0 - float(best_distance) / 20.0)
    second_confidence = max(0.0, 1.0 - float(second_distance) / 20.0)
    confidence_margin = best_confidence - second_confidence
    if route_hit and best_confidence >= 0.70 and confidence_margin >= 0.15:
        state["route_hits"] = int(state.get("route_hits", 0)) + 1
        state["profile"] = best
    elif state.get("known"):
        state["route_hits"] = max(0, int(state.get("route_hits", 0)) - 1)

    # Market evidence is counted from actual observed supply residuals.  The
    # profile stores broad expected bands rather than exact action traces.
    observed = state.get("last_market_supply", {})
    expected_market = best.get("market_bands", {}) if isinstance(best, dict) else {}
    if observed and expected_market:
        matches = 0
        for item in V032_PREMIUM:
            band = expected_market.get(item, {})
            value = float(observed.get(item, 0))
            low, high = float(band.get("low", 0)), float(band.get("high", 10**9))
            matches += int(low <= value <= high)
        if matches >= 2:
            state["market_hits"] = int(state.get("market_hits", 0)) + 1
    # A zero residual is still an observation: it tells us that no positive
    # opponent supply was visible at this checkpoint.  Requiring a positive
    # residual here made the third confidence gate impossible for many valid
    # routes, because town consumption and our own sells often cancel the
    # inventory delta exactly.
    if observed:
        state["market_observations"] = int(state.get("market_observations", 0)) + 1
    state["known"] = bool(int(state.get("route_hits", 0)) >= 2 and
                           int(state.get("market_hits", 0)) >= 2 and
                           int(state.get("market_observations", 0)) >= 3)
    if not state["known"] and int(state.get("route_hits", 0)) <= 0:
        state["profile"] = None
    return state.get("profile")


def _v032_future_event(step, item):
    for target in range(int(step) + 1, min(V032_CUTOFF, len(_ACTIONS))):
        quantity = 0
        for order in _ACTIONS[target].get("market", []) or []:
            if _v032_is_sell(order, item):
                quantity += max(0, _v031_int(order[2]))
        if quantity > 0:
            return target, quantity
    return None, 0


def _v032_price(item, inventory):
    try:
        return float(_market_price(item, max(0, int(inventory))))
    except Exception:
        return 1.0


def _v032_event_revenue(item, inventory, quantity):
    total = 0.0
    inventory = int(inventory)
    for _ in range(max(0, int(quantity))):
        total += _v032_price(item, inventory)
        inventory += 1
    return total, inventory


def _v032_expected_gain(obs, item, now_q, future_q, due, transfer, mode, profile):
    market = _v031_get(obs, "market", {}) or {}
    inventory = _v031_get(market, "inventory", {}) or {}
    base_inventory = _v031_int(inventory.get(item, 10000), 10000)
    current_price = float((_v031_get(market, "prices", {}) or {}).get(item, 1) or 1)
    supplies = (profile or {}).get("supply_forecast", {}).get(item, {})
    gains = []
    for multiplier in (0.75, 1.0, 1.25):
        inv = base_inventory
        control = 0.0
        candidate = 0.0
        control_now, inv = _v032_event_revenue(item, inv, now_q)
        control += control_now
        candidate_now_q = now_q + transfer if mode == "advance" else now_q - transfer
        candidate_now, candidate_inv = _v032_event_revenue(item, base_inventory, candidate_now_q)
        candidate += candidate_now
        predicted = 0.0
        for raw_step in range(int(obs.get("step", 0) or 0) + 1, int(due) + 1):
            bucket = str(raw_step)
            predicted += float(supplies.get(bucket, supplies.get("default", 0))) * multiplier
        inv += int(round(predicted))
        candidate_inv += int(round(predicted))
        # Town/shop consumption makes waiting less harmful; use a small
        # conservative drawdown rather than pretending the market is static.
        inv = max(0, inv - max(0, int(due - int(obs.get("step", 0) or 0)) // 12))
        candidate_inv = max(0, candidate_inv - max(0, int(due - int(obs.get("step", 0) or 0)) // 12))
        control_future, _ = _v032_event_revenue(item, inv, future_q)
        if mode == "advance":
            candidate_future, _ = _v032_event_revenue(item, candidate_inv, future_q - transfer)
        else:
            candidate_future, _ = _v032_event_revenue(item, candidate_inv, future_q + transfer)
        control += control_future
        candidate += candidate_future
        # Penalize carrying stock and uncertain prices.  Advance gets a small
        # penalty for consuming the current inventory buffer.
        storage_penalty = 0.0
        if mode == "delay":
            storage_penalty = transfer * max(1.0, current_price * 0.03)
        elif mode == "advance":
            storage_penalty = transfer * max(0.5, current_price * 0.01)
        gains.append(candidate - control - storage_penalty)
    worst = min(gains)
    # A positive mean is not sufficient: the candidate must remain positive
    # in LOW, NORMAL and HIGH opponent-supply scenarios.
    if worst <= 0:
        return worst
    return worst * 0.6 + sum(gains) / len(gains) * 0.4


def _v032_apply_pending(action, state, step):
    pending = state.get("pending")
    if not pending or int(pending.get("due", -1)) != int(step):
        return action, True
    trial = _v032_copy(action)
    item, quantity = pending["item"], int(pending["quantity"])
    ok = _v032_reduce(trial, item, quantity) if pending["mode"] == "advance" else _v032_add_or_merge(trial, item, quantity)
    if not ok:
        _v032_stat("repayment_failures")
        state["pending"] = None
        return action, False
    _v032_stat("repayment_successes")
    state["pending"] = None
    return trial, True


def _v032_choose_timing(obs, action, state, step, profile):
    if not profile or not state.get("known") or step < 120 or step >= V032_CUTOFF:
        return action
    if state.get("pending"):
        return action
    if _v032_used_storage(obs) > 90:
        _v032_stat("storage_blocked")
        return action
    for item in V032_PREMIUM:
        now_q = _v032_qty(action, item)
        due, future_q = _v032_future_event(step, item)
        if due is None or future_q <= 0:
            continue
        available = _v032_visible_inventory(obs, item)
        candidates = []
        ratios = (1.0, 0.25, 0.50)
        # ADVANCE is allowed only when actual visible stock covers current plus
        # the transferred units.  This prevents selling future production.
        for ratio in ratios:
            transfer = min(V032_MAX_TRANSFER, future_q, max(0, int(round(future_q * ratio))))
            if transfer <= 0 or available < now_q + transfer:
                continue
            if due >= V032_CUTOFF:
                continue
            gain = _v032_expected_gain(obs, item, now_q, future_q, due, transfer, "advance", profile)
            if gain > V032_MIN_GAIN:
                candidates.append((gain, "advance", transfer, due))
        # DELAY is conservative: retain a ten-unit storage reserve and require
        # an existing current SELL, so it never creates a new sale from thin air.
        for ratio in ratios:
            transfer = min(V032_MAX_TRANSFER, now_q, max(0, int(round(now_q * ratio))))
            if transfer <= 0 or now_q <= 0 or future_q <= 0:
                continue
            if _v032_used_storage(obs) + transfer > 90:
                continue
            gain = _v032_expected_gain(obs, item, now_q, future_q, due, transfer, "delay", profile)
            if gain > V032_MIN_GAIN:
                candidates.append((gain, "delay", transfer, due))
        if not candidates:
            continue
        gain, mode, transfer, due = max(candidates, key=lambda value: value[0])
        trial = _v032_copy(action)
        if mode == "advance":
            if not _v032_add_or_merge(trial, item, transfer):
                continue
        else:
            if not _v032_reduce(trial, item, transfer):
                continue
        state["pending"] = {"item": item, "quantity": transfer,
                             "due": due, "mode": mode}
        _v032_stat(f"{mode}_accepted")
        _v032_stat(f"{item}_{mode}_units", transfer)
        return trial
    return action


def _v032_agent(obs, config=None):
    del config
    step = _v031_step(obs)
    state = _v032_reset(obs, step)
    action = _v032_copy(_ACTIONS[step])
    action = _v031_weed_action(obs, action, step)
    action, _ = _v032_apply_pending(action, state, step)
    if (not globals().get("V032_DISABLE_TIMING", False) and
            step < V032_TERMINAL_CUTOFF):
        supply = _v032_market_observation(obs, state, step)
        state["last_market_supply"] = supply
        profile = _v032_match_profile(obs, state, step)
        action = _v032_choose_timing(obs, action, state, step, profile)
    else:
        state["known"] = False
    # The user's requested ordering: timing first, then the original V27
    # price-impact ranking over the final quantities.
    action = _v031_reorder_existing(obs, action)
    action = _v031_align_hands(action, obs)
    state["own_sell_prev"] = {
        item: _v032_qty(action, item) for item in V032_PREMIUM
    }
    if len(action.get("market", []) or []) > V032_MAX_ORDERS:
        action["market"] = action["market"][:V032_MAX_ORDERS]
        _v032_stat("market_overflow_guard")
    return action


def agent(obs, config=None):
    try:
        return _v032_agent(obs, config)
    except Exception:
        _v032_stat("runtime_errors")
        return _v031_agent(obs, config)

V032_ROUTE_NAME = '8c4s'
V032_ROUTE_SHA256 = 'd854c7d6b9f373d31feb57ef1e6cc399c5218bca171fc5801a6a9b14e8f0c09b'
