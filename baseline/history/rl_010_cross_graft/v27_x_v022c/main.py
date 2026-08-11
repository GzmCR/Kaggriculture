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

# Cross-graft: preserve mechanism v27, replace only frozen route v022c.
import base64 as _cross_graft_base64
import copy as _cross_graft_copy
import json as _cross_graft_json
import zlib as _cross_graft_zlib

_CROSS_GRAFT_ROUTE = _cross_graft_json.loads(_cross_graft_zlib.decompress(
    _cross_graft_base64.b85decode('c-rk<O>Z1oa{Mnm^B^`!t;RR5)awyeGZG}t5^I4NEZ{W^80*8>H^cwk(vaQNRT&u(neVkmyYOjxTI{O#{W2pXBR~Dm#lQXKm%sh>my3V;bn*M2UcY(u^SiqbAAfqkzj(O3`1im3=fD2f=YRS9@o#_m<v;%V-=9B!`LjR&eD~w)AMV~<Twc6<dw+3x{cySY`os78{kx0HtHVEh*zaF`{`!ago3}q+T>ftJ_5JtzyN{p$`q|<8ckkc6`swAzlYjd4C*QyRwOxk~5C404+Wa5izW@0BX|q3F-0wep{PhP<|F-Uk`;^01pDy0Le)-3r-W|GqarNt$k1y#xdOPHwU-9Pd<^JgnCoPBX**yRAPk&s-Ea}2=NRA(3r^q|*?>_F|t3JeU=1j!#Df_!a+c#Z~<9F=c(@GNkJ3R1lrLNu%-UXh0xs1`Liw|#q+OC|Zolzg=<*CbLij@fld;6Zp5rr$|4_`JX9%MG1^#MKn)2EA<cZc<I>};6Nr~iK($A@?{#k1p~GPui>p<#X;lhT03eYRdLaTJcfcpQ#S7s;qET|G`PI1}zZd?+u_b+fbGx&0=*$)3hI)XS<#cIL68%U`OVlA&DI$}({IUg?Z!eB3f=r&~l0YGLe_>HGC5Vzef#$m4tBhm!-eGkW=h8)NqLTkrXm1y*_P=EHCB*zL}`;g;gY#M5Tn(*ifGIBhf}N5RXtZ{F-*e*F0l`}ZGTzj^&HUuKIuIJWAzO<Jv54$IMY91nzeTWo%dUbX5G`uAr4j-w-61@<koc{^r%VF8(jm$b!+9k4u4KZUjCXaqlf_1m^)`@m*v)*nVM%ai7HS`|C@bRX5)bz&wI+H?37d{E(D1kX9HoNwzfi&v+f^7NNu52@T_n=9)}pZ9OR039c1<>k%TRfdp~s}vErc{tnE=bCh+d&76p@m5O*O1BSPOZbW_I9h$P#W~f~uz+Xwoy4a*ntJ5H2`-vB+XJtA2_pb&>_{KF_s-ay))fOT(BE)$kY7CFkrtdO+M2O@3c`ASU$^g+d-EUi^8NqI)V=8ZTb14&{c+hk-ZMrBVZC!LkCPi?gR~jmt><~2$31O6E_DpMo@RVZ@T*2!ctpW^IKfwF;gEk!{LuLx>GVr>_BA<p)lGif=H~nsp7!ef+jsWU;GG<U0JCrX>#s%v$f9>F&{Q)pP~^21yas!IL5FQ{w`>uhZ5(2O$@>w&J8UpzrrO<#jWESrh$r)0vw=lx^)2=N-TOa<Yi0Q|pA~j148&){syJ=S@pB4_efaqP?*8}t_wWB4sYR_|zjLpmx9lF|>3P~S6n0)?)yA20_{GT}NCWe<5rNoX4)fS}MoivDKK15Ujd5<$SW&33SYylKV)a~xkwv;ZejD*Atk>-9W_*>Dy|KW9ON)k>BXsUqSf>Fi(LDhkK8!l$<G9+&rb%h-6%*PUvbAZSftlQ}Nw?il2IAIIJcLS~xtI|+1q6{O-SKON>{ZT?-83O3;R>^`ZY_3z97SUs3C%=UrxgUoW3#cIx|Vq|KxZN)26L$a#6&J`IO@8ad@wcym~0^S;Ui-4E}{;>9Fta3fB`45$+OP+EG6F-cxzxLIw&Ba_$B)W-3}YR?<{;B4CrxNAz+Nqs@zvnyex9rkIi}fW>P1QC{7+uyFC1r1at!20v$}i-kb#HfhRAQ;*<x+*%*P);$s_(%QHPToLtWvSlrWZuM-37YLpSvM(1|FJ?ONX9K1Ud!bHbt==_U=1t3mMcYtS%!ZqzF-c``d&Y%=f)8lqV;YfM?KArR;s#zGX!ybS8_Va(9yIVZ+&VeoA(}UyVmcB6Cc95<FK$lK76+mn_<k}1nZbRV{XjapMm;e))6=$8qdTQ~O%s8w4Kr_lZixUOrr;%EMS~x%kdd_o7KKmK3E&`&m04-MN@K#?}$(3H^?gCi-21hS$aO=zYu`C<(e8e0?PHi!RV1L`x*!f+L#Mx-1CBVQ5l$LRAr$Bc1B=*G9YldlAU?M`k_6GUZJy)|I!Jp%lpr}bds!8VFfi7S?fSDNcEqky(Vz6a1LH>Xlm^yeZG<fIm#b9Jd1z<XMMBZ{0&YGGAX0P}mN?<aATY#q9DNa=@yoGslhSB2nge9VbOy(r@*s+x2u?vtjI%M({@qhzQsnOsD0Mj@hO#5RBQ82!#GmM0tw?SP^_zoJel88bAlZi%8AlZ;Os}8}^a4WQtJXZE)m62B=y<8W9vQpkd*@4TlkU^@l9{?`FqqP#j6P6~>VyeYgVA{;-x6KIiu1c5EWS-?{i~Tt6T)3>B*8UH#-~90la8`j`q8+<^Rc=^jaii^y-(}P}uz#~YkSp)lV8vGev6gil-C7QthPpfpK_^4+z0}@TW)|V6o&Ic;KR{^<aaBp;r{+9ZG4(E`FhoaJtkrd-6fUKqoRu7ZcbtdqxYvx})hZ18fw~rP6grr7<^o2u84Va{^4cSxzeXTDDrU<G>5S`5QDTMA1A`47r|KlT8M7i$`|JA(ATjK=wetJAwA!TmOzV!R`K7EMHA=t%*~G+FFNADzKzQopa7DkwQU0@*hBZqx3+&h?d0|ETu>$ANsLRp_z!(a48q3EvL#VlsCumf>1fyR$kRxtWTfs4Y+xH(!Szmx6bJ(G$zgx#|xHGw9UyCThmK@`^)_S|!sM-_Ak|6Fcu#dKG<mp$NtDs331#j@p)zgqzW|0JQ`+7iIhDpv~^KG%?dgRCJ3i<IqzgObLZD*`os01)!scY*bYK<|i*&+^Js)+MQgOa*zuU`MN<#EC-kUK$X)TWRi#8@XX#^T%K*4kGVV9v-nK3A?u>T`AU4FS(<(;Axmm2B7&7|qU7613XP8Zi_{U0)fzJjEOF*vUsU#ti@sadi+_qM%t~2FO_akB8f9^ujn66cM2NXjYI{It3sm1~wnim9M2z#j!a~_<C^ygWQe1gHE(F5{b?-YFO%YG+Y}(9b?U+V9&;JZG{U$yGU@3k5TleCAs#-j+j+Mok5TmtaR2$bL_%q*;$&<RyzM#1CV|u!8D(-YY*057DF#9KlrNS1r`I1r<Xa&z8r{eL>oqB1)Na?MVK>&X%nLw4c)eDgML*Sn`p~KuH)2%wFrX(H`D_IJ1v6QCcsos!xd+9fZPYrNtz96j~JS{vc?ZO8`X>BEYHf(rTIISyaYpe^(ju_Y=GUTv|JI#OD0Mv(Iie2^hk3UPme-OM#|%(myJv%A<Dkc0v@^e&lSn3YbDZH8L|_VAXb{wl%+vb0vXVXQ9dHw=L3ftL6vg?tNi5YY|42BE0X{)_vo4jtfHEAMN_xwnTDJs%2YbpB`ZEnp4gl3r(KDff&<UYu*}2#UE4a<RZ3RY{1%|8Tc%&w961(5m{NAiHi1y_+=UuTo0H{u0v<bf{E$e%tyk&Q&Q3+GI;QK$=8i1#=My5EgURh8$*UPt^dCm&=tVL|FTfE~*(kDqP=bzEgxnQPEK-9B*h(>^lr&(f+t*4565JT|@9GazEW+sEt<gb7FM{+zWs8w4b$1>f!+2J<Sa29wB3&Vfp_tHqgpon#X~47w|5YNBUivy9Ci{u1kl`pD)hL|cS;Qd3!lX4732h@`)Fp^|!)$wNGQywnAv!(=8&MFw20mO8$Vu6^{fH-$$q<7|18?kz=#L*m3exbn5tB5MbCu|&l_japd>RIi@QvO-9FXz7@3@<#4?ngxdFHs<_&kG?Y7knp4M=o(A!S860f?U!5$#6r3z1$XHJ<COR^^ny>B|RzHD-(kO{TGsz^!H+`mjv`GfjqbQ_a;lr@`EEb1il;4QwMoF*@^HX2MBk^+*h$fwB+=#QmcPqQn!t;>S;C*e$^muBXSp2hjyI$1`clRW0Y!h4LNHcrMA~gd?U9^;d2@feCwcoq<6gKM)ENi2U0kaMgMeCs{l?&q(N_EVCoXxTk+Tzx;J<PNYuzGH-x(-B9MB3Mv+dxlvJy2tJ8)O@SN9us$>6W)xFRE2w3qN}?_9>*h4Hq-d~M`O__~r4JvnggMSlCz)HtiMG-k(rh^u>>MnSMs6K&vV&=JHPo8+Hg`I7fuKAoyE1U`a9MfDk;cn|+Nv1NM8Pp87vnIV+ogGSTAB<<&OD`+i1I`{GFjZ7R?cYw%phBurhE?9k|H(oPAJGY8Nrk3qL8N!)=Lsjbc4)P6Y8X@^>9wMGsqiv84+-*N-Pn_tK_tSoMJYTKU^9&s?f`9q=rsxKl6aeUDO3;YmWQ%nNyr+5dktsbaHyJPX<^tdBC!L9=&*jd{UqSu>50|AKvhUvIOU0PC86#alG={+hij{$b8J`jOMxW{ir|VU`l>En;Zj|`b~+^8WW1wH=70!#AFmf)()lq#WeXD`e2kzl>w#FDW6QRS3E+RJws>|%V+UJli<eInZ(-CDX1Gnqbq9w4J3@fP)vE+Pl`ntp3jm{?jm15_g)oKWP=qUvEU-|5}4aEEq3PdpyCb++{;{F<u?+iwh8Gl!UHo{6XCSVOvcNBy^z8WWjA`X3=HQ6$f&*J-j%!(HgePh;U1lT8L+{kq4^R$zUkz~U@CeA2>a5O%u66=?PE9-SnUg(lTY2(X=dD$AY;3J##U>V%nCziRyg$Bz7>y`mUc^g4YW2&Uabl*vusb4jMh-qh(0WgeSz0vuzi+_6$|Mf8xwO$GPGO9s;Rb{;Pzg(%hx~^;!Hf;W!d!^pjOsDurnChy0$n%&B>>s*Wu~N{Fd^cic?JmOUumg;u^s^<qbw_Ou^#CB`rL!kVi{xe?Op2^Y-=*;kPG7Oag6;N)CXTVrs3Qn&<Na6a_tzW9SIQ#4U>2iwGCoOEqN&hV(rk$O<xaS&j@nJgnS|G_Tz#TqO<{QGzL+rSKNA<=cq;pPi0(M0^w)9uAo!anJ`;!Fj%vr!miR4=NmiW|D*_E`kR=2XSgD^sJhZ4-dIvPEEp#0~eh&hR_C#cnJ|p+WOX0>{SaNkx-CW%$z1NCpmCfJiff1+V+r<;$92MON+1dnkKL(E6yLq%e!@_x)Jv20K;ONuIsJ-<3-n3w)ZnC-ErXEbV|53g6o~~u9kD}>u`0G-L=t8g8z1cJX4YNOQV<K{=!Lw2oC@#!MbQ<^IsQ=gxZ;#1q;B=f0$4VnTP>{v_s$r>4=6h?FlJ`JJUJr(GvS}7pu)ip&4nM;oy0!WL>JZxju)kpVArGR6<U99<87e!`{pl#;e!15|hLX5b)BftTQ4==oUWq=m0I+KS?fOi=k6=i0WE!ioE$rpOLW*l*r&J+YXHax6K*tBhm&~Iq7CXy_*hT!Yn7epc1bk5O<Jo#a8Otwuy4=Cf1!8(InD~1}nKXjt&?NvYX!7mam*mxg~pu;vAuXaT>~=8aekCfX8(|8)y6+9<3=urd&>13+pSmv8_-#mn*y}smHNI&l*Sk^h_y;!B}bDAdIhIPO29pLLmcQB{+5Az@)74f$B*zog28;Jx)A2K2kNF8ml%7ICQni++5MoB?{0Vq?%35+;LK^qHh-><E}yOx@gxrdc`UTv47*XGn7#E<Hk(a$Edxuue)v}D9i?i1<f!|!pOx}933y6*ZT^MT*tj+VNtWZZZe5^_2Tgib9~$3!~tLbxIc=t!h)c*;@EQpbSvRT&t>N2hw|dR6_8BiL`Q!Ojh2`FT=wxzBk@}M7)dz<;Ef%kkCWusEb6zx&{wO$=35WDj<F1}n~?2M2iiH%!F!C9YT!kh2~1jsmrOLMH5N86V6f8$MKv=R8Qt~w40E|@kQ|*0`*Y`VpZewNJ7FEzK9jc&7%ropHE>Hu3GUlRDwVbrETcjtBm8$&@QmNplN=#R@$%r7l2V#AKUJ+ZM??<zz@}3*U}j5q7WTqV6m*LXO`^+l<?v!Jfiant92=s$5|L0!64qVG3-pXeZ;h}h98($(C8X`O%~(l6q{5~Oj<O}y>G+OPAa)hu5_ACPQJEMEp$KlCRjUF!?HYytX=vu+Lejg({SCL)GL(?1UMuUs7|GRvkGfS!NbTOyr;c|N^(dJ<U7;v9L95n_c>RBJb*O6e_ynDi;Dzyi#$us*3q>V^0bYeNBY?w1@i31h0_t!mk0l~>o1)87*pRyY$&uklIuREZ%h^RAAFYoALc-zojnCMm-e)@qU=P%{Pj)eBq>Vj!fC`%g=2aIt=y<U>-K;9LlX~PuH99(la`m7R1KoiomDSJIv~<UC-2f1p9cY-X);^7~|6;aOz~2}7L9-wpZKtc}{W^{Xjz3ESW0?R2!_^WZVDCbE(J8xLDOR_erd$9$(Imm!cJHdnbYH8oAWQQtW^4=4y-(PWncJo+ILoi6%FeT4F)DSfsjQxEx67O-y7r{fA(f(z9v^~YYKE>D04X98Gw4hppa!vADoIALo--H;8uT>tnDJBgRvo2r?iM^R8JHpK3}zaNb4VB<b}Q45xDr3}Q$Yguq&$+uzp@T;rgW8izT>qhDKARKqYYtz61R$Ef@!Tzn&H$c?Lid)QVNclrBNb_b)^MO<w)KZwm$*fLeYKCN2blD8Rx96j2bdVIHJkqloLxkL$1+*8zV9(%=~yttMHQHOeAQhfp3BoPVE;G+Z>Q0h;U}LX#{7T+c2gp5z|`0CFLk3td$2$|GdO&`5Fk!)vi)={Tq;E99C+)F67nvWdJEPCC}}vNnq=^d-R?dG{!U~^#l5uy2tJ1i=j)Dpa%S8RqFvV;=5;*7Sffl@YJ0%Z+nu=f#s*uBoU@)*pk4Aa91A&G_x*vty>RR1NLDJ4;=YRF!=f$Vuigr`jobdTL{>b3ihms2FnE3j&4o2U3jaGzc+zA|J^7H<J7lPf^b>x$yikx=vpYppT#jc`ApQmrtyA1$rJa9NuEtnX3Yq1aIF3O%N+Er4YM~Yu3T)Zs0Gz{V3}X_0;PjX(mOWrkgA1P$+@EyQ#~J+GP+v1|7^f$VhEG;=YQWQrU#m|uhbZDkc3i$A_#Nd20^#g4v0I|38`o(!<<r_!4U8G_Q)7lyRVSWV*L(*x@{~VmwI{p#ci^bZkySo3e@lt?NR*DOgSP14ah;@K?e$fbc~O9d*YiW`>gB>=NW$V>?Y6kVqvNx8ex}RQ`Vz2Dm!Qn&#;I>hb;6@sM<s2Ogy4H8&RZoK|{E+Wa3Dw3S+52Kd7b1KHhZ6bK*K;M5NCK>UNd(PT8K+=MX6#XA0kwy%_f!AxUh3;nLwix$T0UG;#s-ccpMMsVY7<_$kLj`@fz*gk4qFEV*6ab3$mA3f($oYLk<%036WtG?CILv35o58B)owsHb3F8&T0ALJ=K|gZku@Y0+sC1EN9bmMOjhNyEtTBcc?UE6TY6x5l}nQdAfdB>oY7NK?eje;>%V029V49eSLbxK_dZvL=A+3)Wo}{TU$zu)G>n86$RXUI+5B0_R{j<Psgv&#tVQ2gc8wZ!-P|11To!>>HYBa##wLDEG-?Q+;cwfzZ+BrIX6#D40{?$$*&fONA4{>dD_b#A#-cS94^FnGcT%qZMICA(0?VbXzlIlp^9}d9o59!iww3nWp6@I-~_b=ka;@T<ye932HDgt|~wttLumkI{|gDT>2S#PD%nL^BEE;Lvapls-{}<0F7Sm9tJdvX)YJzhUA0Alpi<^t%hX6bAeSOXbX4%g@j(t19%xeoP{!vE6^+b7cO_966Gt%YL@pvF&AAak%Xp2+8?Zhv}8RdJRE~aQCk|6NT&f9V00<A(I{$_@?Hvn3@z7Q)zoS6J6si8HLlE2?|{OwoTIoi{UXOaE$G;4M>AW%nfy(R1`#a)Mv-VO2Hv;`eNlM}cC_;mO)#<+sg#WoKtTX$16DaD!_0K7GC@_95Y*~A8}~Q{=@P=&5W#;YG(FMiiOnYpCqUT;sTG+_4@U}jS7hXT4Ru)wT{Trw&d1W>_zzFgS#gnGgbj8}G08#{xIo%r`XkMpCLf(*ektmMP&uHsi7)e7R*H0a11v<$I1)8?HTk;I(L(<?LBgL0QzJ!J<!#5qy^2FdQUkCtXep#N|AT?M&HA~P)PbB#k!9ZQ9o$SH0*_~i=&nTL70pUj8QrTc#Hayo9VC?l)kk^>`Lf`wU{6Sr0{5l{imh*OMi|a4X#r$L3<8Y`L=2QJMtg=5cpJclQ>nL~YwO<eMiLd&hr{rP8<v}(w+@_auVOT`moPq23jO)1=$=e(226o`YR+Kd1z!2m6dQ~uM$>FLRC=@$a#Ua|g%ivUR@h}i!)Gt0kpz@P^Tj&Ia>Cy6S})6a$vil&JyoKp;ggmppI+K!%vs4Jt5y&Lmx=@@v$#6Q)lx~l*=Vs;w+2&=9)pa0n4N|MCQBkSWbaGh$*)<p)3wyhvx-UUuuKaJN@V1_(>c7IQAx^4+{7!5Z9Plpc~H=2cP6^q_1Q5)Bt2fQG4;(W{bc;Mgv@4_*JfL-C70Yv?V}rhOla7qq@{^+a%g>?P_GL{4%iS;h#^X{{!=A|k*%x|V%M%RNAZmj5Q6h0CXebB_FZ*_gV$5#=4OjkQci36mp1$p=MOVo^4zgb<Ug{l-~v&5g7PS`X)ejD30%}yVpoN*XujaI>OL^`K#>uWB4`$6F2X$7LF<?3)*{|&T^%h-nA0reZ)76Hb4M}V=N$nWG2(3Oebw7$5>VniQm%YLyE7rUvG$r*krLgCgbRX|!DaCnxqe~OM?5Yz+~N)zlEF7g;+TdaJL*XV(x{hFjHHP@A$TwYH{}_?3mEp&*TD1Q{7vNVUXWl-xR~OwTabDRX-vdCUFCP8-kz&>&l_t-3iOa2kKJZ47vfQ8@(O8NSe>i;{UCTK1-Cs-_9_bJX<io=%{L$cwfoK(yRXA8J6~>O4#ESCsBm3Z36cyUp!tlEkdcB)wLKmQ9ya@}6hdxR+rF-_k<|St{g#A#;mboe#KhHbHp_q4L`^C_l5}JDRnh4gYCTM}+zR*iIG9-ebU&!e1&z9`J5??x1^XljBSZoD*j*5Y0h;chyzpMdM6Y(O40MS<S{jwUrgj5-y(*PZudO4oEl>YBfu|$Z2iE4KUb6g4v2kUVs#8j-i6L+{A?I^pk^0#E$6dtj72ls0s&Vn05ox9;&fGekt0aXkUJLvPaNJlja#Rofc;5JmxAgU#0mPL24ug5$Or|0lw~DZ{IvGMbj@n}`65*Z|tRpcIpL(1oW||LuV~jf_Sl5u{M2+F$U9SpOAYn<-S`D3=4}DNrlTULMX`yjLw7XuUj<lJ}T(dT-62}BN=?v+(=Cy}*jAi3}dxf)JR2Vi>^{u}OI+8#k88PbSp&(V%Qi<|V6JJEwuCK+CeONBn{U8vstLO>|Wevw~*!RVEubXL?ivB}O4Q{)6){wJp)HB`%lROy8u>>pQa6Px{v%AKOjipe3H^K+H&QDH9%CD=EYsCwX@kW_-01Fh7S;q*=E3OE5NueTN);>PAL@_E1qN|T$33d#CKU|@RtEI%*qQ@iGk*dWGfT?F=pi<AOdjC#Z{Lr>e!U+J<B*hrpBTra#Jh<&*RLmze)Mrw8A7-SPRnBRG4g;tv+LP|laPN_ZsN}tNeLY>3cqq^eKrgKa;X)ljoEr_>GnFtJaxa5z`4XzT_|Z+EfG{blfUBuP`dqdCv}d_c_>v{OWs52zxf^W78175=#`G=p0IbGR->kR2UY$<I?`(SCmF9=4>U&X+ONieSDX{KxD`!D{__C{(Tl)6w=}&f|&)GzhCgwVcDsGpd=qp})-?c<Kgqo*U?4iyWFb-zm{*@4rcW>@~T;7PksfzcsY*dq78e69_6NE)INQN;|z%sF>&=c3J_S1E!_({n-v*m87MY*UQgwD?Zv@93ma6y89imjqLtri)`Q@aS~Ol(TL(1l7#1fG+<V}jw(hFM=~MU`$rS|(9>(3S2ko2Cck`d1L7h|ndC;p$^|B%3qaoS3HvmTJDm0NRx9{O#JhEFTDPo+9kZom!O@z-Qa!SDA@3)Q)y|%U7i)Qy)M2=B(sp(@?molu~t!8K`y<4+%&k%m1vLbYq;IKmlc6s`xIjnqsXoZ9$L*r}R$y?5DXmY??dmUaIrdRuowdB14>u$4j$<FwVr20o%pBTcr*%B=&R2ElYnniNv=9#|$8dh(5C%gjvq+M$__j_#v!|@ii*U#n(x3#pDMRS;P?sYBiPPq~0NJWPP!Y2-t3MgCf^Rb~###Tqg}BZ^cy!KrrKq^0Lf0m5nVPYFbe0TkOze_%5DA=g~tyM+H9#^ZDXP{b%}`M&HroG(Ta&>R$KX{PKNv=|4E6JulBB^xIX73}V_eh5&^)J+`@-%-3nMJnNhF)ytfuOuD*5Wn-+nsF*@jlQJc+dety*QOC;!Mu;kc?j4!s>j&S(pwh~0QF|+6j*ndtU#etT`q`kfGHXvgl!BNyF&hC;XMo@}vKJ|)C?W$|YDaTlk}=B?bPL&HLR_U=e+{8Gd0j(li_%B+CCd8>$fA&(1*5t}OA;YtB^WN2WRl#dwnefEU4?I{+8kZ1`#H2hTCtFErwwGaQjbFEaCz;__mQgZ&`VE^0o8ja0PkW1luzm_55UxqEbH*z)U6Wgd@Qx9UvjpSqU{yfi^v6+tz4(wm00Xdj3}Zn1U0J`3XF}HNCcH60&ofvA_$?FIM&CX9eJxoeQ2NIwKJo`-+=nh)w61$xn;5Vd0aHYLP=5_C3&2NHT@}aR*P9|zWkA*cY&NYfEV~rsazZMM#cxL>g7GDhQnn`%E5S6{7rQOFbygt3fmGFnX&B^)$kJ`Xae1A9UiZWiR$jzOV!!QOde;^D%mt%EJi8KmLpQL*obUJ5P5Cse$!gF(1CML4toK390E=lC}5gS1crq&)H2qdp5l^QA;iu$wH8sSG6A(76iq3Kyq5@j6pEKsYD_7ap=z-W0X514VlV+epRB-FP3<*Npe_kyDerRBU1y#bh1Dw*G%9gBh9XK*i_q8ITl!J}V$cbaWMth^e=xBVpzT?g6Fy-FA49?t;sA;98<7pX&IUXNTsQ@(Sf(p4`v}LF7G6u2g{(H<$f$z_l!ze#_6Yroz#lD@<~YhxZ#cp@@f%X#BP)}|8Mz{D5(3oZO-s`g_WD+M%Kkz85I{6>6b@i9>-zM!*BSP5ZKrL$sD|BHBCT-3T$5YfW`tyIys%PonA(NJ(jB0%F$uzw@axbNP=AWL-?Xva>9I9~nZSL;*z#vd6e20tWN4LgqxBhO&{k$r6sYP2p=jzId-YVC$#+T=PZin5y(qMb(mP92qr}H4?9*)_cNmoW18wGZ<X<W~6>63{!h|F2<O(N>QWcPhEW3lYmcxrmkz7ws0}9`W6UR8~V#AJor|kk7CR81BYlV}vu~k&)fKfejOD@HPk{B(G*BOYY0+M!Cyr?b8c;tGS5iX(;8ao$-L0RYEK9}9Qwk1LHmano&Lz`pt*w*shE={&$ub&lKT=!u}cfjSWo15;3%D|Jsc0Odgs8p}{cwn|gSV#v4KP<!pL*eHzF;}IYJp3QSe5Pd')
).decode("utf-8"))
_ACTIONS = _cross_graft_copy.deepcopy(_CROSS_GRAFT_ROUTE)
_LEGACY_ACTIONS = _cross_graft_copy.deepcopy(_CROSS_GRAFT_ROUTE)
_REBALANCE_ACTIONS = _cross_graft_copy.deepcopy(_CROSS_GRAFT_ROUTE)
_trace_actor_action = _cross_graft_copy.deepcopy(_CROSS_GRAFT_ROUTE)
unit_actions = _cross_graft_copy.deepcopy(_CROSS_GRAFT_ROUTE)
_CROSS_GRAFT_MECHANISM = 'v27'
_CROSS_GRAFT_ROUTE_NAME = 'v022c'
_CROSS_GRAFT_ROUTE_SHA256 = 'c234e990fd63a168535b55de1f11289fa2bbc563b390390b49d0d65169cedb18'
