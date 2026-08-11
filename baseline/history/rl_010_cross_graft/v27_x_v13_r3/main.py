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

# Cross-graft: preserve mechanism v27, replace only frozen route v13_r3.
import base64 as _cross_graft_base64
import copy as _cross_graft_copy
import json as _cross_graft_json
import zlib as _cross_graft_zlib

_CROSS_GRAFT_ROUTE = _cross_graft_json.loads(_cross_graft_zlib.decompress(
    _cross_graft_base64.b85decode('c-rk<O>Z2@k^L_`^Pv79Mft{&+LmCBC{WZkyo1JI0NXII@E&IOw%Gq}jYw8iSG;)fA~LH*jeKj6-Bpp1QCacv;>Az@clP&Re*Nd)em(ocPiG&lKYlzroS*&Um;e6j|9t+#=a2vV<=6lE+y8$4{L|UncXzwb|J6SH@aZo<U%!9%<Mqwi`Ps*}yWNMg^R@ZM>)ZY0&mVWYH=qBwf4jTBKRbUp`}2>xo7?wi=d1PM@c-vWQonos=T9FdR~zL2>1@CIc>hJ7_qTWVZ@+wcT;$|;Q}G^taJ=x}g!piG`{vW@`%ye2#t)y~-Msnv^VRP^ebK>0it*-5jN!uL_oi~pSABE+diS_!{buH$<PMLzn_POnM0gALOXOBWcf$^TUhw--|HmqP)WyR_HtO%`J`eWx#U`%rcX!8k{NrynIhE@1+bMO9*Bux6bc5H|kIH-eQYYn&iyH1Ue8-x8xB|N;Kv&ivW<TS*baVqzd)6RgH9lQ0slLGy8q`NkZLkF0)aKU}wKiHp7iHlGb-v(8Yx8%KsI|$TbhVjVb<!4CgRc?uugSwzP!>>#uOs1sBug<LI;qHhaFo_f?wPK-$$k9c^p|}+OB@Fe`Z*id-5S1-x}Ncy9uLr_Ys`<<uO&x8zvdcGF4ga1F}v&bjp-rB>)V@~-Rt|G|G2xme|PilKaXEtl`DR{{nWlq{l$87cl%-4r|IMF=C{yoBJvo)En*Pi3AAdw-m`h)nBvQpld;=gHvuti(wfv9Lt%G$Rv?ZX=Q};U%;>D^*PEYjN7q9;U_30S((&PNG_^W}0m>){@PDmO*KlvE)X@pEO6|JrCjG}sNF0v26hW+n%&kd4SK9kt%LZY~ce-wHk}R}vHzMkE@3|8ImpgoT`1W$Q{ti~nU*t+GyqFHit$&{?D1`RU_0D~-|1Di@=HG5J{_R%vZ@Htp#nm*$vr>v;j~7$2j?94qx0v5vh?G*UYVwwC>N=_-)x7<AmbAC7Pyoc-%Gv)ax3o&MD*`o1c+ggzcyh<W5;Jcy_FAvskmxi_!S_hJiT7)X3O5~J+KCrgLLiep`3f>TJEefa=6A0XaOnP9DZQ#%&r*bMx-huZW#w9-=O;UF|1KW%g$F$A<3UdYwB9~6#c?g<L=Q-pCMTLeof?(`yy`fGxSaOvB1f2z;vglG;|x01kR=z~K`Cw#mb+OjK|cKN?e*P%sE+VP$do=h|9t5>sAdok-UG$6bK|b!4z2j3EDEH}s%H9o957?VAh`?jrOagpbxBd4kPK%^n(u$6-a7ti`UzY*5}Ks45sVO5vIJxnfnYw}Z@TGrCGhFX>;OR&dIdW9*|S<adIGE>$32U67kc1kVZb6DAsxg2awah>09<*#rtp-8)%xU`sr5VC8LvE@k$rl_TsZ66`0Wfh&uTKK=0PbqOoc>Z>Q(S?k(6L)i&A2SQG**hr{qk7A%&lKNxzg}wyYHjmehGgfo?xUD6X3mYd9E_3-&TlEk|=TWd2@$(h}NVw1@Wa*Oz@m_f7wZ{bX@^w-T3qMJyVW_fZgAP(?pXZls`&U^J3B>5K@<1&UqK895l1-NDl>JKpV0qH<!F&5}s^8M4V0ft80blNC{bbBWN!kHU%qDp;c?^j6k?v9gl1d%X5snNo?*8)`{OU8Epl`vv7z)RGc5sff8ZauJ5Kx3@Q68syJ?rwhE8RB!9%`u&@HZ+{%8&D-~}dxN|K(JOp4tMh!kzq{W5u)Dka%h~x=`~s$3?|!joxhl;ZbTl5fKBGah_uq?J@%6@&Fmn$@)8m!F|DFUH3LdlXEUj&?$=uIlg>M<A_u=C6LL4eOOl-V$4?uSddNub&$pU``Xadwo22Css8%K&h5`-D^d<C7P6ayEJE#vrP3k<0RifxQ;*vH|+yj~g<T^f9Fd22yK5;_8fT>6%PdJ|h}wT2Gj>_#SpV=0i)*T&ieG7I~9LBJfv+Kao(t<5+IS%JS!-l?$K86PAab%*Re)Mo-sl@L_{AF@cTt1$f#ccNq+k(lS*lIFec)`X!vKS>Td4MO7_XVb`Jn+TnzM>l+#iyUJOsrS{3`0X&VX4wvYEO_#h;PGD_#G<Hkr(q4i_iU5w)j<yi(47Xc>9;D-x6IZWX0-YHU^jPKfNWiaBpV9?8N07mn!r^N<)ZunEN(<(V4ijXE^eU>=bmPWJc<-87lmGAU&cI|Wmyha;7)c5aaAC@j`fZ4OI=h7^3>*e2ar4mQRVQPh3!Pzxeh^<V9yCO(Z^ecd1BRLB<MQlxuI}SA0XabJelC$NDH5f0M1o|z}YPU{UH7j=*j-Ct)Rr&^$@nUB7sd+cxS8GA&AhF=sPQ7!T2!sob@Wl$O@uJ&n1gm+KrhkVODug3dXPNS|+^t!?m;E_I~##QSy)PZvOlmCP)OX=Cl+s1Cd&DIxY<>e|Du$ukcMNO63II(<t?2K1yAyQR=EGN`3fH18rFngqGlRe!0|pZ^kc-1E*XpTC4!h`ig>VeLqP^xxf}Ro~m?OFzggQ#w*pf3b;yo5JUwoz>V#;k&X1LEq=UF09xBvCe{yV^pa!XN~)3GnumvEQ!CySun4<_VLehT7Uz8%t?HQBV7;cm3aOzLGPi;ONIL?*a;P%#QEbeLleD9~8u#=&QdO&(3H`2=6wrA^$Z?mWe2E>3GF#GCbl$HfW|Uq+5EuSrGAvLdtJev?AOg3GlnK9mH9bSEG?VL}Hv~tlFrmb28}!EuJ#@oK=yfKBvrtG}VFcBSg}|E3WgMiS6~_i~6dn#%K<tZTl_(@^ylel$QNiV%vTpmJ5xY||1hNX<<M3L$SC+WJuT3n`o&=NIz}CegUr*#HQXudc(cb!07!8m^;44wV=xLS0&u#j#hw+WK0V@2yWS<E{qPY3YK0)%PFQgOfz7K}P553`_=nViA^$2ujQy$hnmP1X3wQqv=7#Aa1Wt%qnu|0|cJOh?XnhSZUw@OnCs{c?3pq1GNE^n;5sm7>W<EQah!lH&nERn@0N@sF<lY|Y9!%)woU=L^z*pj@5b2gfy_UwhD%3AKvgP5#7_@KmT+3cZ^w#hueVN+?kZ0QZUrB5iiS5&H8vM~9umdy*N2>Y^Bp42)Kqq)H#S%mPkyu}ax7oBrer0}%Zx(-KSZ3DdRR7c|j49){T@(Uq_bqKLQ-{JN8`59OCqGt<dk@7;}@U&;fZ!KgIsI(xo{>q93Dg_G*2HK<>!-BH(LUi+rfVQCG9k-1ISR}A6zvrwEr8~l2Yi(|bo{|mb4QjQHZX{2<iR@$8NY-GiBbaH-L2Bxw&P?MR7+l=7M|5iHvr{M9wGjJly(D@<)!K?PNew@LMV{#SxN5m>J^oMz?i2pczIhRl(u`5#Dy)|3aG{1z0RB`_61qWCsC9V2`ze%8LA*xScwX#&WzhzX%5{iNUp?p~qzv36f!V>cu0vfi46HBR<-a^s;q%dhA&jRF0S*-4taT?esA9eX$K6&R1_?f)L#>E@<78RJAZ+hKXr%&SFh#zMYl^b61WewrW?p59Hl_~fL9fGIkStR3Q+X}mHmKrDA%`LwwLNCz^2bcr1)orc3_8)Cf_QYO+(Z_(F>D+wP&yk(_X%a=&fC7!&anVU2&DRRw%S?n6fa!rL);6qECqs_H23;!Rxi2`C8XO-E{e%piySJE&aJkUrg<r&)f=K@lU}13sLfk$)2U`8?0XXwM14?1pJXiC919RCK$6y#or%(sE}25-oe8~1MzPQWrpfG_wW9#5(voYsDDw~Gys=pG4XM;iJ*Bi)aPeAy6Pebc=e%kY?g?(C6Sd)1^2o<h9mzRB(1^f9mQWhwBN)ygBVvx=fj?iBT@1nsG+rk&Ld$w{V7q9(Y_#=a8@tiHH!w<^Ko-uNOT~?fPzinO**oRV!NXFe5FiW%WDgY1Y70!rgFDn5WVZK7RtDuT2*6Dy;7#`J6LrmSK`>bz4-Kn~uSv&u^sVy+>vaW?OQO=!NS^7{UIC(kZRge#skvkw_k8e~E<G3ovQb<XPtrtdMgh*fP!lX(vt`EC`zt#D;t)!byVqY3lK#zWfNDQ;Dl6KD{nyquy!3QU_ga@calopxKxzG&{K<tRG;=^vL6TsH%$O8%Tgs=slgSAretHHaYFs|+3P+{|^1?s?^KUhCUHSbPA){d70HF8_{=oSKkHX~1b6FJYpc$ofX@6(%BW=KMrSDzc{hbONZ_$N9@q)?W^1|q_1ETSKjM+~~AcMumaJ!YI9y*c<#)us?5?zq8_0(jl(N5QKzFz_&Ta|j{1VmS*qLfhO>BxV!j{E7`=*^~a(%fTTY!0K_{@$m-o%YJkf`LATqPWtLQR=!j{nP7I?^E_+=zXfwj*aHD?}1xy87_?4VEux@yhzyzy9-=?5=@x*@&toum@2x$m}vt8b3J3LXc{?UU%}S%jBO%;Z)NNiyd=?30=ztASfWuzlbwb=8<!bIeO9|)4qypXu~!K?w41WFRC3Q|jyiUeL>vh=E^;No`zP!^)`uBIMHP<p7yZ&BY?QF0vUi<GW7#cbB{hB7tzP_8H<!QHg2Hl@1+m7EN4P<Gn+p*rE3C5gCKC3o^@<aT*;>|+#<2!avBj$&Y9WB-i~H<2y(>#<vrHaC0rsw>s3~6B;(LmM3jwImZSnDo{fjakI(bh}yJ4#WUbYmCbuaK_RY{wJI+lVwI-=ky2MX4w&j<9uv^2u&)vQAoNxDQ)h4=kR?SJDx2VhQu>>d%~kn;yT57Mxprtu>i30TLZ)oUnPgA1wlbiw4&0;|TIEx@)4g%(x|!`Quyh+3T1Y$>`m$fJ+G62mjAx`pI&OABw`)tQPHAW1vDz!9ZeGf`MSe)1w{aih&ef-HIGKC<!jy=rTLvReAhwv}yrd+1V9T!&p`OVe%B(D&#S_$0a+?hKYU{my2&ilt3L3@3^|9;kCQlkmg9RGI{*Lp`HfDUNJ4$kbTj6|iH6J#>yxBG2&6dC)ifdHIqmr5L7piMTT=Q{Jn=>Nzv9003UTE{KC%5cM5f!}MpOL1g4*tyTUsUuqfdNKr0xE=J;?C~e)ofen9?XqE#_M*$x3c<15YTN~%d#N}GTteWg1xxw6=fCM_%$uogb58e}~7yU9lb%uoqb}ME(Ed>;>obaK^rm$2BV(fC<1v$4N2xz*8=2y81Qp&c(9B@F&+UQ(L$_|biF{T#dGbdGH)kf*jLv^`rgbfA_GOIfQlucm9kyg@8c2PWh`nJ1mrz-jSIabN}C&)v$OeNfjFXQrZ@1m%A3uou(PM%k^rqu!!aOPgox@n(kw+nA`hci+9>Itgb!2k^{PO6l%wHFp0h7-ldovPPlc#XqhoTyv^&s^yV!xDrOZ*~onhiTX1juuhCE0vX9Qua>AjrY-366r>AE8J6^LX=@1h;}em_OwD~j^u?Evy$DB1C4eOqSri<6x2s!(5TtaAb#XWl*oB2Qe!mG9cx|ic$aNRF`Ek11?x*0w#jr3R>7A89><aI(m*JWe;|4;RK3L=v=D1sa*+BR2VG7l*Iac<^+N1|mGAh;ON_kEJanugq4sVrHH{l)4#<W=y|z+~LQ7~b>Q1kAX)3#V3+Tlu0OsXB&jn%S@+{|E0XMEmmW$ABjiqrZcP*8BTL&VS8ZDENI8A9qOYhweyK{%Z#xtG8b_PM@OevWS-=(P>iiKQgmbU~P;Cy(@eWjEf3$^wdi$TRkh0?-zZ%<UK=It1uJS|67UdgcUG#%Dk2p&unVJLYMw?T8c8NuX?<((8>Jjd8f7T*+LKaA=NnWgMV3vh{E^Eu?fv|@J#bZ`TPUu_w6Mko+0I)MvAO^3~%BDIS?yoyaa;Z#CBsb5L5e@{XgiO;8iw$+DCGmVCbpn|NdHcNB!k{WX{wPKpkgeo(&xX06@m&2;I(7?`k0$ey@qfMNuCmt?}zBwCq<MF2&5iooFheXNnp=LGHOvN^qw$SOXBxdr)5{;Bk;~I_?0m1nH#xS&E7|1WE+<MI$mq2K1Ic)*c!g{@0WFJ=&54-H2Rl&%vLi5T(83<@H*zj3`v<c*J3`qw46c}BQ&5AT(9y~@xqGsBz-qe*W%@M*C>L^p==4^_H2uiV*!qmv<>r61^*Zy#vmsekjeOGI9r|SUUask;z5RQV&2$iSRz-yCDWO%<0IF8Znsh!65r0@X-zpQ8yWKs1HDSlq4I{oR%T_SsWKBBMTSB|i>NM?|t3K2vu(3{gsR0g&<CDA^mb_O7im8>F2Qq93m+_(38DmeLsK{^0`MW8NdFnX>WG;k1a7SHJPrzVhtSDG7F5C_75TE;+U<ZOZJkioz40NR?A1hK>NMkJMyO<Y`d1#GM?X)pe?VgK}P_ytP9^@$S&J~V+h6I`kY2MP``SCsN5tAaYpz!1{5E`nHs7KU{*45kVWAq0EERNOv&F_o2AE4JVY2mK5}_PArVK5DDZ$<9zByd4C2+Mw^n4<EW>N?2QqH8R#Utnfnay9_biCJfH&5sgwt&`~?SgihEcksLNKVn<#l$2;s4o;;f2$Wl&rB|>KwOmj5^xI(wi2KUQxibHC900QEChu%-(U{0iW&~SJv6Q0Nw8JeXPo)9ABdXxwg1Z`nK&kO-JNYnGw(b@tVh~Co@?a}e?;x~6~I3k;svRACxU0!*jpWX(gZGgzL2$`*+izP5$=E?~%p^@7e8`%w2HEd=mshvM&K-)g!s*<uniH!oG2ZTD}=Zuv+S#E=Wa&cj`o>R}!moPC=21oifU1WmC1I>@e8<WIi@vcazsi$>sY^Eu<5oE4kM{W?Z6oL`jXBc7aYg-c3hB>W6G>-f3*o5$>@b<;?1K_N`Ism8KD721oez``Rj`KKZN0$bK5&qO|7>CU`z&Ko!(jg>SU^oY$QxuscsM&O$&H*eS1ORA)woDm7zLzxvRh_gf+l1x`xY=Ohh^sV83Mz3ZL-#<<hG2@F$f1G%pRkE^AB7BGfzU(L=ofTTC7?iAguq^}p$FfnSu=Wy9r960(Vl<n4*pJ2TVi7TnD#TO>&s7I(9;6j&4J^!Ih0v4XbT=*N3>!k+6h;AanPp+Hjuk;Jdh@T*vp|G-4xu+fW{4)zLcJdu%SqTL3dn&(^6oyZ_znmSPXGYq=o5SbTS1@z!+xbv$PzNyT!=WcQ?21kEt-cpwNS02!$34ErD-h`?dM(PW&gk$2QLbm$z9R8c<jf8ZJDKL6?p;{zfW>K5p$Ab<yZM<|JkHUM}`<<yO`(^G0Vf18P_Nmg*j9Y_OXST~ju-<CIOm6+G4EdFFjq351D{tzr{idO@TvI;DjKZd(68L8@!>)gV-~Ylx^NkD*GKQ6Ef|)MrFJ)?ABTNUc&G;iIIiz#1L@mUY0uee(<EUCqviNqOiDZP=#YK~&=)*&Lb%*Ct!-Xt|z-tu^Ra22Vss+cHAXxY)-jR~s-z7l*c4xsYNChL4nMhD?*`W2G+zrW|d4t;`Ut;*VxV&enI8uv2Lt)r3jByMFG2Zy;RF)gdt~hBN0xt%>xLi)IT^ebqU?>g3T71R?TW>+mmvVAIY!36>!hWQ6Q%iDW%!7fzhvjL{6{e=<cpG}TxdW(+M!C8J``f_=!k^m%%=9X75pW;yC`_+<?JK~bP2`~>z3E75vcwk1>740)I2{#5uQCAXq2MX1czx?3Ynhp8}%0^HNic|uS{DK@tU-4yYuK*K+8fBc7rbs$y>E_+t74KaG$#nBo{6XG|KAU!~?j`Dd&P_2h7?`DI^tQ#|V7bnT&J$DKv4WU)e)HP+{0HqR5&iKiuYzJHuT+24W1ruTFoqA2?K**6pFhHA}xk+Im*Ak1^vy9Fqh*sbo*xU)$FKH2^^VmvoEd<kxotwb9W}QTG7|GCE)CAc>DFfrYC6uRi%7bMB7lpAzHWB&=+YX+hH}C%Bd9VuC32F04nRbO{*6eVCxQ#5M!u#Z)1f}U`Wqns(cu(8Sp0)%85iF;~lWdtG$edbFtN^Q$wuqinhnc>g_9R?OfZa<74ze;)3WhC(_rJ+aGaW*G%GH@NoW1-g)}lgSp(0fngI_Jmo(up^6_}>Q!u8-wCcIgZXpA}wpC9d$qgJmZ3`_&|i}8R(fbuNcR?6UjWB7CghG~%zXK4JEEJCe;GJ5{LiUChAaqhz-25L`>tWu-NmSd0VHu;N841x)Rt^1#gR`}DJPa8WaLzL|%O2ePl&w^Kpgy?K4Oc!`l+fq`Xk`&V_?PgRkOl^cpeEcTJ&1h|eY%E!pak&FG6q(n>&@!~OPvA#ho*gPR4oybC5DS(HKZyZRGV_x*!FZn=c+CDsadql?S8Z@DmL!cLH;H{*Fw}Ju#wUcl(`RhI^E0@4I;Ey*U|LTfdoYEjjc9jSM%(Mbsf-BAu|qkOHz*o>^yo!$`^r26V1q1<h%}x<nN*UOsU##5$3gti{4^j8p;EKeH6c=>Rj_0sKTNdBfi_|<hoTGSUWUD(K8}IWh12?zfoQz2Mf}V9<j}|jb!bYmfwMwC#Z&xI_bJQv=4du4A5{gF@^sz_1ZL=gvE7jxftdMce#97jpa>RQ0;2`kTjeudE$M5W`uf?yd@&q9<cPI9NlJkgp0L93%P{Ro!xp?LcAYsUE2o3FjW>Qv0aNh&eH5cTGkE#+S|2wj08IJlq2^x5>P)&8Je8EzOZU1h2c!b|IGaw%1E~~}*~VE?tidpks1nL`HDr}!5P@)s*)|!Mw>5>!5eOzeoE+cA3+Q7Tv>=O30kCrTV5+FiC&70tF^7|0#h^$?V{tdJkrJ5Y8o7m27Zh*PU`#r>D`hkWGF$r&fi;9Io0+(7Ca95cCnB$6V!bD&GEOA1vn{uVC~8UI?sZg|h1nN|CFxnwIAH*x4QPOy@$=Cp>T5tPUL3h;Ww8by!6<ii#7i&;j|NdOnn-R|DM$_`PDTbTfcX4k0D`8)m<pOyI#hU>)rMbj7C~+a$+(t+h(KJLg?KHlf4tcg2B!1!fgcypU#|{5HzjiF8RwcjZ&F{}WqCd!Fm)YK#l3G!mem8aXII8%5bk{zw!L!B)<mBrZo2YdtuYADLCh?H3j2K}e|Q$CH}LD5B)eLLSfg)>%PPGR-@90fxk`e1)z`5M(u;NSF_;%8jY&XJmNF}rmz3U$l-@}noyHFO5=j*wT}L;_LNDUH59hd+F<z`okR_+IM62G&7!ts#L;bb_@QLq>G%r&GIEg8i29B|aElA*qBYVQ4RENaBD>0Qaj6E}epv>Tu`hG6q6qbrZT0+_z8;o$uvHml3P$J(a7d;5ZNXk^zII5)aiPCY$5}R2QEKQ%o1Azr(hRx8Y$4{_bF6J}sen?&O3=l-;pLzI}mi&}aY|LsFl?MGRf%{A~LK~wNL<C}yF2WKJ;EmIzJJ>l^QLd9|AEea44`&}z`w-BxQGuOlg_`E)2(%RDErb;bV7OrXAe%D@a*z2;nK}={PB7Ft{sj;^$)#oH0<NZTr{nBIYktkbsiqmXS72Zurk-!iv|W8C1oLTHsxp;$M^F_6B4zStkWs61F0GK;7}}^ibVn+EhK(Fn#9>NPyHmjvBRg6%VIUvN5eaSLGRRec=U_%cGh(UWsA()A46%56E9i|%5G$8xlHti>K@Pc42mnk%z*~x{D?tJBgbo42<{vkWc0o|`<+lAGxf&ML226EQH?6HFw$nS)5I5rQ;3Zr}K~~^R11a2So@gyA6OR9`_G(f6Qw%D{6hiGY;6o#`HpFp7y?9Xoms5L&xS2qIXrepF7*Vc0Cc7RPb7q8S4kb(B#c>ssxT5SbbWHE1GGIb!F(iIGs>Z*FUH-s>NYF5|EQ?H%G-JFfTNo)%5^PKj&!eMbwl=ul)@cB;eRvRMW)CkyI!3oq?cqDdizq%L3wpex&TKODJi;x_VF#Tdeagb^=`%=P>Z=^L6T_988NbfE4?yIJsm;nhJXc1F)!NIOCDCEu2MR2ok!f!6>kQ#a%D8BDN$G6SK(ouGl$6m<br}mWsMXcjfj<|X_0(s`nNF22V*O;4V<|)3F16=bT4M(sLYc6{_Q;m`Eh%UdDgINK9q@H?Hp}BH-+;AYhzj>u5y4_r$8-LAUm9i9%4P`-<$|@}FvgP867FG=>=R^tpOoc~?|ekiT`~xy9MsDc<b|U{gn|&VMzNPDilb~GH3dEbqL!&aYQ>KYO@Nf<b?w^}Z63ow+K@+@ayP|>3^ICAdyzJ{i=jZ&H8&2ClXGGe5|2L3Vtp~Ho5m;#4gp${&!MBI$SBN+WnonquM(awK6h@PWhDpXb0)(Pke9`%GFvpzdf?b<U>b^zW}vh`M`PAvUMP@pG@b<;6sOIVlU8FBL}-24Y+3H<zF0Z8x^PsyZOWq9{YcDQVg#fHmQ?M2729Ab8LFBfv)ag=B5@spiAfLhy?>J1%yDwBn$Fo;9Nnc*1<f-TNn%7qYSV)Ozpe3i9}3<&ISIa7HH?r;dYog1KYv|dBb<Acl0!E7n(%du1kb(1)(d-9JnuVuaaXY!wr1$ljq+wVy$<WTL<tYid)=>9XC4Gvq%7!`<SB;(7G)!dLJC)sC<VWlR7VMesrz(XR6NyCs+<g4*Y(pmC~CFDZ~gk{@jm<umrnXe')
).decode("utf-8"))
_ACTIONS = _cross_graft_copy.deepcopy(_CROSS_GRAFT_ROUTE)
_LEGACY_ACTIONS = _cross_graft_copy.deepcopy(_CROSS_GRAFT_ROUTE)
_REBALANCE_ACTIONS = _cross_graft_copy.deepcopy(_CROSS_GRAFT_ROUTE)
_trace_actor_action = _cross_graft_copy.deepcopy(_CROSS_GRAFT_ROUTE)
unit_actions = _cross_graft_copy.deepcopy(_CROSS_GRAFT_ROUTE)
_CROSS_GRAFT_MECHANISM = 'v27'
_CROSS_GRAFT_ROUTE_NAME = 'v13_r3'
_CROSS_GRAFT_ROUTE_SHA256 = '9d46e73be2ab901ca6fd0384221b8f42e2407cefe40cde01ab3130371b217462'
