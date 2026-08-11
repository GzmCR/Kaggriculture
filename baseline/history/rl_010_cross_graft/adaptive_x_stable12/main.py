"""Adaptive replay controller for Kaggriculture.

A complete season route handles capital, labor, farming, and planned sales.
Runtime logic stays narrow: actor-local WEED repair, demand-aware SELL-slot
ranking, near-clone premium preemption with exact quantity repayment, and
terminal liquidation. When a near clone is detected, the controller searches
three turns ahead first, then falls back to two turns and one while repaying
exactly the shifted quantity on its original due turn.
"""
import base64
import copy
import json
import math
import zlib


_ACTIONS = json.loads(zlib.decompress(base64.b85decode('c-rk<O>bM*5&bV(b74}HEO)2aOe{pP3`s7L8bT1DDGC(nBJHl|f3IRu<l~!}GiPS*eWcv1Oj9J^{l4>Y=A6&}Ir-bqzy12lZzq5HeDd+~?(XDacJlY1|M}N{J-+ey^4Fh#`^PW;etiA;<o(s{>hZ7Ki*G*t_|xTws~;|}Pi7}?Z`LQXg?Rh!{c81b@Q1tA>fPhp>-Ve6`;*!0(c3?)u5Uh^%;wvVf4seU_v!8Z?i*(h5C5I*_2=sC_n$uQo-`ly?eoccb$9=ztsib~@85rUwS8;!Vt*j+R@c|Nr_Rl%Za*-5>h`b0LAknq_tWFx-+$VS9@h?a5JYn}KcO{YH!Sueb7KG=y7|g!|DKP3ebAcSlq-`zerx#b@mybDzFloca_<qcZ`wn}EAX&yhx=oHa5v5PeNFxTTaW+$aKGI%`a6*)zr7p};HWK+Lv?w-x*ffGe(3H~qh_Fy9i2tnMhsiNy1X-<9{T0o56WTMK4Sag?&gy-T=EIZLf^J{`@?OAYrH0!kD6Elvi-_4pI+o9e%D?!W0gUZ$Isv}D2>);)iBdM8-6-5UTCq&&Dmz)#s^`C5hT`|d<R@3nRSPTFLN$z-WjxW_m1_b+yj)s+Wu+u$Yc+8?G-=#^dj)P=%c{A0$+Pxh0JHIi#BjWqL1EOU$5TY|MczZ_WtVn>MvhLt#ucsXwTTl10Q`p&;Dq7x#^Lu^2bM^N4s%i22(Iw+U{(?-`sp)3H{B;P7i(C_7iG0Km0c9l;L$Bvk`g?Q=|=Ym>PJlEeA=u;=D~H&c4{A?GfJDw{9Dh0Y*(|HN-n7$!nl8z=1JRhB)A9c3;EN{b)77gC&qKJIC&G(#xCr$puq8a&;x{rPwjs;1bGQJw4W8?lvx*efblw%T<y--h%gr_Z;^qOB`Sw@7~ae`3Jdu)24>ZJ(?z6V`2CIYx+vo<ruvvxf)D9Wt@GB+uE~UUP|nT3CG3x!ER>yteF>RM^_7Xk`ZEme|h`o_?<O2@ztpPrYY$-iDE_u#{^mLyWbusGBS6c5lAjgn_O1g$x2V07O!_h?S&cC&Zv@qt95{P=b+E4t+s+?Gn}m55AMAipFcTQAj9ZeCaFWNl<3(yNiz4!tY|7nVJ_`tcIEknGNZ*`v=TQ}6J&F#EzZ6y1J5jx7Q(#0ms#-{(&ygM_?T1s*zc~s>tz1ZJFJnfS&5qw2cl)_bc~XP#_Xwqu_4pik%J<&aH-hQ`<ng6sa(yMmcuG}f>U^X5%0r3_79rE0@m_zq_6@hl41^PXV9fGXjBR;6W;{xN88YF*^Bm<G2t?YXQ+jH)@q76+6QASXEK#l!>+Z-Wvy>+9{-uYLjK6_gZ5}mzAKa;Y`mDe``gR)H>=y*A0MCj#>99m9x^VQV$h7dF1C&$aYri^1~P0YU2FM*K3Nt-uz1{Nqg@iaD%%yn=ksQS9;Z|Q%z>x-_6P2I@aYZxH3K=$jnK2dH`ronlTm<vwI_3>7LgmtUN=e=LU;(22~Eu-u-lYaIF6K~i=@}W(pm&cf{Z~;FE?qlfU1^VTP2cz^yFL|RCCHN$7diMr(oWKoz+m*$ht+s)7JpC<cwEf_BtAHN|Awf*qiyoq<OA3OPx_;nA}}pyj{wlDLF;Aq1t@9jR-i_#(vZOiVpoik(#uc;8?fVdij7HsQ&3Td&tv~g*Gr<SVJjVt^t@5@9*eqt=guP@~BgGASeuW^yUFyPWmLkV_7E3g|)=l`2&y8-uLC^*)nDOIFm(nvb-u=uPiTUY0!}7IXk27ZL42<MZ^(;AmH<_;)x7X&11`2DWgx=Q(|#Q`xu}Xx>`%y7sxL66ONm87%+@f1V}B;uhp5Z*m7w(Yr}QHn|Z|gaOQ?t9P0=)$*o+OQD7zXgw1!<n6I{nBjy0%UOLBwJ3zAe5NgJQpeX3!-T_3H`ogUsELyW%IW{c#6y|?g%{3@U$y^n*7B&SIWMx2_{4QnR{cv^t{f5nFL>lwji1eR_t-x5geC`LPP2vjP@Z0PD+_6>1(5e}GVaf|k+K#&YS|h$e2LyG@*$pWMh+%-1G-T_cv7Mn}+`tTKvUPaawSLVs!&HEIa|uBxeZ-!Y8)na5x-_#+Hl95*64XlOmrl+qRqh2uvZO>+ch4=E{j?dLT^A*jaNBRi>Cm{8c$Z7|d-p7U^366Ry9u(zR6P;jNw92t*!B`eu6uSldmmMDgLO;9A+zWvd75n>_66)as{*OEKWD5~`CNsesm<I<^K}ppkWtJMZiY9WYw^<Hlk$f9WDf$sp9|?BoK_tH)+1N|0L`4E2#qplKKTX9B3EjH6xFBZT;#k5ru&H+(4aSwXay>PBfSY57}Fz)&0307?75ywMnj~QBSz{>8@OWn;OMPkm{X~5CTnG>AS!6BQ-9VA$%9Y~Yi5)gj^RR3g!Zjgz{TNsQwtuTvXj<$^8~h0dNcvfMzy2yq6}}Irt1J)D=3i^JPjH(Ut1yivQ@SpcC9R)<6O%Z51I6VHsC1M;J0wWk1fpnXk+*8_U0ppoYu{~p`n%@R~pA9R=4!T(4e;+Y5;GrU5o8_Z(oS{K?A*5Mh5z{O`DMq`qYi++BmHY^MGz6*Ji(tAuyycQW0w4c13{pwHZ@2-Nz*PqHc3xT-*BcRO<pG$fXF0!;z`_VZHwrBUaExVc~3s&p)hOTAVh(w840X@<+cSXXapbjR3QY51cQ}m6t-*4FyofZ!XMF#`<poTLF}%08e)Dd`s=kh54q^MI3dJX%K*@DS`*3=m4&9aS%OJM&OX(L@~LbS`Gj!8T7tp3`eGqgG6qpRFRK}&!2}SXCdZo!%sel(#{gG;CWaV(FRp0k(V+pN|gZxmYx|xB4%jWX!jX!Q+g+Mz=&Cm!Bxz|l~GtEblBh^84C4M7=|Jc$VLW%0#^*nVefP$!CfHICiWtjHaJOMg-~`iDgj=4=Pvs0TqQjoT$eMB(XmvcjUs4DX(g!KK?s$=<v)5>Y;S5msHEpn7JW6<Le2C~lcpM754*!Bgc%4gxe^#y_9NmL)29kp!0oxfqGJ#Gm0o0Bdl%bCpnFcrpb5HUEL2%*vA4d=p~<zrwmrPk&JL?<vMmy-7;U-{+Qs2YfiydUqJ+|hp_o`T!7vzhp%)J@fp_Hq*4k!+5!#T{0e{yv-b+xMh;eN<!2=IuT6WsX>UA0!u1C%1pFAwHjuN5@4)ROPE-{EiUvi-nwbFnBeM}heV4*UB^;IGO76XQhCW}#owNp7OM#smtv?;c<k0rqOUaKH=h62jtmUQWq2$}~l=h2>A(X8V5Qqjmj7~Br~!f0ATzd##L?MzD6Iiqr0Zvq41swCly6~GmiubS5{5n~cQK&HMq_r%to-2YWUf=oj1!xUbB1;~WM_&>3#Go%};IYY~b0K3>OtN<H|DnbeBwi;Wh94<SZ2o;zzm3#z44CTS0Th__Nq$z1>W|^?t2T3=B1Ku16nsEm)on2#oL}JayaS<%|!9HPXyTCT>5Z4dO3BZr`gu4K8RRTCbAumYEm{c3jPD?(+i!P<si$69MIliROs%Oub{P3E3ta>&$%$cPhDbR%~lqqPF%HK?S34D}-B;(2>KqwHXbE)gI%T=2^J6Z^N9##a}BjB;x`6d|zm)qI%ch_y@>15&vS1h`92i5kl-B%_1eE+$<U%v7=5~g>pmD-U&wZ?S8iz&hZp7Vcx7<PhbGmje{vqd|Hn#nFUb(ups#e%#>#7pj)Q*3Ge2Z(d~WD`Bp-pD4}KKZoXY5)cvF})(rs7yI>vgFM&*3s1!@Yd4erIm$-gC`*C*`Rbblb_@W4BBQVO?!EIcSs)hEs(kh_aI9p*fgeZ3@h{EJx1}o1_jC+MzB;AK$e$-q~;KZUzN6o=3ksqu?s-6oWnVE3;uY}*O}*hvgyx_Jm;Hk>}wNNtPkjyzt+5GGq|U2Do|vgEpnWZTp}=@iQ?U2(v!sV;nY*3Jm-<1{@{|WjXz^wQEq6L3|plTIPEE7zoU72Q`*gxC)s|mTP=1%A*FwoGXmQS3WvlOd?TLW$kJ8Lf+lHV#%c5%CAhxBujtGr0rOCnU9kLhMO5ae<nJI3)Q7ZY${uSx6p)FI;xcHHlcX|{YL-gL{z;1Zz%Ey~fn`F7y|xhRHb{fD-RUTyB~zD51uLLOD_9A@1=1So<uqf7)>0-jNh;ULX+Eyjl(LkQ;+Z#Y@%*Sg`>aEMe0f#n8$$L}{mVK23kQNeoi=ByHuu!<NBB~uZkb+0=Y|xUk0l?B+W&&H#{2WR`TqW@YL?b*pKTt%^UbC+!0}^CQm;7<@}cYMP2H5)ZDosq24<My4we<i$VMn~UJ_nNu#~g=A50Cit-xpaie#+Z>&KV4{QKUk?2053_uBd7Zh5ecl;fb}3PX}ed5I7G<z7oAQ{<~0O7$>fcBN2K%Lpk=gBD03NhKUm5$u+y3MMaA8q$4<NXIB%*8w(WwsV2J9p(1Qu!$k}0kGnMR43zZQWA74iC(EKh!!o+aVjwlYt_OkR}{LSr$tgshqjo%YF9meR7eXGCI5mVHp~kv9pFV_79C2_DYkv$R*ov(14D3l;A2|qHtn;WU7guv_%6UG5(3!C1`|GQfXWzqctJY+uE8PYh3Cj}#e<f&RY0=y@t`hdJ;ZW|;b{~zrKHi;1XWOMO5(c%?%;UB-ag=aQdZYV204Z*xJiz%=sqfQkZ>W}R@;NYM?@@+YGA04x#4DTK}{joTsz6>^=%(``YtGsC{G1I#__aQikcnP>l~^*F>FT0fB=jS$|NCmoRVC-%g+X&Fv{>V_hcE5Cu0hB(@+2mRa^5CEePzQ0+W%pCJHBzd>F~t5m$~6@HhyeQ;KR4jfeq0#(Ur0@AFqR8u+?WA+m+?NR&NsPBU(>fAZ-vj(Z`7vxCJ-K)h#R^l0%CC;}j&fZ-<1212p~<>dzV<u2!%*N23l1+^F1>|<s<$GPWHXZlg1EBCY>-HJe?Go<Oz_eKdp4pNmXy~_GMB)@y-r5oOkoHv=Jkr#^?sl~M+1=K^vm7K8aU9i#vOdUc}X;1AwMThOow(B`SU<h-K!72%THvP>G$@tmWaaOf03jiW2$q<EHX5U79YsDGnDqM`TeI>>Q$tRF&rL;5Pd&)b+1-2+L8tD)@xGo@Q{p1(CF<hYVwW9bWBU5?4*C_<s01z6`l~c=`d#K<wr2nnmCJ0|rL=UL&ea7=2E#_h~YNBI>G4aC_idVk()~@X2D5>$yMLg`}>C5EwLxEcWWk}TlbD#y?&;oh5sEgUMUW7@3HgB=Ks&pq$T0_v-IlR*jQ@ClAI~ab9cA(6vm5DQ@YT9WuemfL{xNcSGju$}5!A`ZV!JJIVCCwMOa9alpp<;A@3M~*Rb4iL_Q%5fF%~L_hGztP!#wZQ}!U&`$Y2E<%kgBa9UD!4=l)yrOat!ErVoXYbBZI<5WVa1I3Y5IQH!q5bdG-)MQZQg2SW6k5he+^9@tk42Ar6DXqxwu}7YHo{Y(QmEBgBLru=5F+HiM|nKB#s?Vh<#kiW}S+yEBo-flt;-3qGR7j?geUzg*3!D^J$Y2pr;yF;7F@uqw&Q6w1gdolfmaGXM7*vMQ6uIC9uB>~gxSA|GBFkH^u4S!{|6o8Lns8Nx&#It)+y(isoOn4k94NTg7PNG*szdn%PGAr6gZE$+mH%7l3Wn27bNMZzAWxpyD&i!O5%6K2L2CMY|zP$p}`VcI4_9e`4MVd9Q{iUurH3grt4k@i4FIIwNq7HEMpJ?hJ|SumRLjH{Q0-zaYo8#J^68N$tK&3=144`PNU^W7-c-5BPhpf7T+8dHxRm@+St-avN+VxIV8D!{xDCi93>=5&B7IrtU0rqv(JAsHb=CucdeTC=F=B0{c<$$XT`B?N(_XmemNcmMo1f1^}sD6b#s_$RrKQOFdgtrYH%1UgFfRIF&rOjp;5%p3UhK>z1Cc5BBEbw_@6ee*$62Fx&LY0*<9C_^s4?EvojCdSs?yX8(Z!mbkRWL;-8wa<!SV9FIripDD0)Z|i?{a2!dILDGv+VzltD8lEm*x=dJVvfjyF|~W1;8x30EDr^NQL*3}a6GlHA=~V|g@l^T7lt_N-k2?q9gwz%p?oFUzxetPX`puAN({#-juR?K1U(s7_tKESCc@Dg;)<w2)iB0On5CGl#wpL3rqZktFDuK&<M=}*=e~`|spos@_8IC|GgO-gkMcJSU<EY1Zye5JOp{CnUev<_{Q|WNZTvM|D6QzSK~xl8cwLx$$OT6t+AybQ3b6w*UOZHdL*7apR00sSG%+R&fQ{Dy&XYwo-Z~n#P`XsHo#PPksYe+|lEz%bxcC%ub5WW~S$dF<Kv{0*fx2xiK!G7{a*r#(NnQ<6j<pq#&FKj80%N9i8FR)7f)cI%3OJmcR8y0dL5^NbzY=DHx$r$uWuMCjI|w!qp^z?pW?bbJp$xFyh#;CRR=&8;(87ECRNN#6eB1uy(|ENd3kvZmku*Yl5m|sP`3*q`WCw(nMm%VOSkHc+nvgpmGZ~PLh(k=HK`7kWK}eQizfrQrgaQUG%t#gE+Gl17b|G`chvcqoumnb@K}K4t)knbO8g)9f?uSGO^0KpK+9f@x1p$>*G3B)7gC1=RNjC)7aYmt=!=2@neQ>1KPc2VdI7o1>vHQ+&pc5;w#RiL$LZNb{btHRkeWp>QW8F}Sor4ADq?EEGJYFarQ_66hycLOOuYKUy;DR>M;f}1k2<Bw?bl>bikTq>bwOL891GKCd&N;;p8JR^!GK_h^Q9890RJ|%JFem<|+^8^LkwxM)T5S^WsXG0Di<^VSDKQYofJ(5<6Vyp+LdscNF~pDmp#tJu;pTP@0uZKPU`sL0t`ZS=RZEQeP$pd2TULyjlMd2j!(iGQ875o2yoypsfNX&xi7zi2he#ADk?^dJ+5EhxkQpiDkn2vvg3)M$Il1;})y|?+O%>kqDP^AY?!UYGE>r;97d2AXfAT8|QL@~N%uO5I$dI-_&!=z7X{#WshsD9sDtc0S`dCb*cE~(;?)hK&t>?ca%xgQw2zKX+^yF|~164QG>6GL<I*5&Sz)_y%c{Z>v9S%5}`OZOdo22WJw85Eb*OUN*C+DW$Q9Rl&P=zidLEz$)m?MYf+dGRu&Cxz&Y1Tj6!Q8r}ty4Lf@{%VSmWh>igBiw_$5jDv>lhi*g5~w>)U73N8ZykDWB-@5a*EZhfNGR24EPYB*s6+CH_<~z1(wn`NQQ4Se!HPlrHIoyeausdgd1Hvgy@SIJ--p;i4^4Cu0{O~pHGWFOO=_VHyLE7WBLh@{MMgRwIVpejSqZJpNMALwF2s~BLtL{s-QLqwLnUHin>A;dn|`PraP$nLsXm1n76-&T&;(Me#9Fc&v(qFoBMcO)foR85P(rSx}^*832F#lbkq=J&SJ9PbGnm#bq7JaHS;IJb9ZQQLJt4OhylMC2}5{G;Ud%Z3Kp3jf!1NzLhu=&4y}vmLCQQvI<5OLwez}=;wloCaY|#sj}evU6R9l5rQOQ}c7ErXxK-x(aN%ekJ~AN4R$+kj91-S7cTyw)q&w(|#gt{&)YfUbsG}pv^rPJ`S`LmRhJytLJO&6Qo|Rwrp|b$}R$%7~Rck{z3x%!AyBsvoE6H4d_D4j1LFSS)W;M+p(F7uyNM495)fue}W=JXUQ*s7Ki4ouaD1yXHknV4vJ!JSGZJ%)-I<grOMM{??G*ikUW?Ol7WZNqRmXXczNa|TVp54d;0BS*TP-v}&$1wI5#77eG7lPFQwO=W4wGR$+NQ!-sfo)I*j-q@}ij)jTEf%Q|-G*81l>!m8&XE@trCYwt2pkiNW9TKc;M&<k7tv2fWjNaWu9~0Nb!gM#yyhpCVI*}j*%00#B|l-fi{po*%($xLCoV{H;&LtOS|`-|)C~n6gL*WwpI*w6Ls2qljss|YDkp7%yRQ<23vC<+ezIm<S$}&=XQsY25SH8D3^7Gw;6>A`o*ikl!UGw3(B$MAw;?)&kbrv-xy=Xcb}*79zo$`Ohhi-w>K~bE=Jh?W;wB1;A$5b;iPB8Mfn)S{h#WEnu5R!eT|!>91y^#NB<s^C>YQi-<U`iz+H%3r94Q;d#6a?`_<i20u#Ir>W5MxO4U^bFsv~jaQ>z9m*o;tX^FXH8GDtE%Gs+I<7snnYNwLizksjv~t7AeGlKq1wM4~?tnTCGW+PU!RP7JcP=}IM!T0Rph79a|T40BUNA2USS!@!}AU+j{^21l6D4yRg4MGPBBxwX`&q{tuG_*Nn!npB;{NHQ;P-jejHpOPdsA<&l-bJV8fw<;bPU2_Qnphm?JuG2&^|C1=cByAC3{c<bHD^yBOC^ZISBx%+nA?>=x?7Bs<seD1>dKi^I0$GS)NBP`MNJdkIhlGL~ijnOB3Q%;Jfutz;xWLkoqpk!5V1=UcSOPi`)$HV*P6kKAP){)cjxppW*J35!qHVqqah6%2RUK4Ny%i(HavlE(r3G_=icFlws?XY>Tok_evlpcnxcusB&a^b)ir_|d0WyA8<+(0+6xmt!(HhKlG{}42U48H{{U{3(Vw3|c-o`gIA8R?7M4`v<#ft$3wD6|YF67urS)M8Ly=Vxvu3JcF@=*$rdB>n`=`L@G=FrRXM+>%^RnldH9bTOk;DBI)3lr4U(QN0O@>^7cGvtSBK6G3s6blPr2vFuAL9rZ#A)I4MgryCAjt3<!A^|h8jpS(Pdh8_WLxOOfbf`STFaz{?R6msgh&T(`=Pu78Oi)3gesTkR6oMw@Re1qY$~L%m?l`I_g}7iuhX>CGl1}CHArkZRj4ZX-3kFK{&d54OK^g~+6ur{rHL<HCB#%}gb_xd33ZM!a(E%g!{229z_984&q;6-n$~RS2DYv0gdTdkuM^&*M;RBQ|iQxnlCMGXNt$YP)E3C!tvZ94Vc~p!Z48*Wd$4Sviz}{7j3_kndMg&dSe#Z;oI(Eu>=7oPiG(c4O@S|?!XKR9%ol|Q(4n;=r>R4qL+0v~-Hx8hrXt*{9sJcq1B?5NINdysBWZ_g`^L3Ci^oqIQ-<wo5(7J?F@m8tsB3VhqdRrhi7g~`>oW+-NAOw6vrgT{I4~87T@tJH@7!fA}UyPb!E((wotdyB>9HsZnT7~c2wq0cQ9VAFcAC4915D4j+19M3YhinQFkSF53sG&w=tp#^TO!N%6VEx)b`hC5d(FHwn3~gYg+)M<>v(IzU8v4;+%q>!U#Gl;g(no)D&&p&Ri{axxM7zM1kZg``e5(2J>`+0?xHUz58yGIkmkga7!WM`bDP*3M@FEiOXhE@Du8=2#PH|Viv)Tf10xpdhF^r)WN^)A0dLW%9*DpEpBk<{JMbA%0l9CLaPcr4LG{A!g26#s^ZS&nysFv=K;UbNl1agCMXTYC|@?l8cGg$VT5HeLF;ar0-hw{qn322hk(ZiQB$#DyXN@MzXmXd@Qu+!QMTdM2>nU~{wVM|U);(>EN7A>2x26=g$#g<t<zay_D0$<^ooL@Amr<7?#Nz2N0a{O`6g&B2-P@S{{g*+@Wpc5Un-h~&O>YL@^e?UVC3;')).decode("utf-8"))
__version__ = "adaptive-preempt-3x2x1"

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
_SELLABLE = tuple(_MARKET_PARAMS)
_LIQUIDATION_ORDER = (
    "CARROT", "EGG", "FERTILIZER", "MELON", "MILK",
    "STRAWBERRY", "TOMATO", "WHEAT", "WOOL",
)
_WEED_STATE = {0: {}, 1: {}}
_WEED_REPLAY_STEPS = 8
_SHIFT_STATE = {
    0: {"last_step": -1, "due_step": -1, "due": {}},
    1: {"last_step": -1, "due_step": -1, "due": {}},
}
_PREEMPT_ENABLED = True
_PREEMPT_FRACTION = 2.0
_PREEMPT_MAX_BATCH = 30
_PREEMPT_MAX_CLONE_DISTANCE = 6
_PREEMPT_MIN_PRICE_RATIO = 0.0
_PREEMPT_MIN_FUTURE_QUANTITY = 4
_PREEMPT_START = 120
_PREEMPT_STOP = 680
_PREMIUM = ("STRAWBERRY", "MELON", "MILK", "WOOL")


def _get(value, key, default=None):
    if isinstance(value, dict):
        return value.get(key, default)
    getter = getattr(value, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(value, key, default)


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


def _shed_access(size):
    half = size // 2
    return {
        (half - 1, half - 1), (half, half - 1),
        (half - 1, half), (half, half),
    }


def _projected_shed(obs, action):
    farm = _farm(obs, _seat(obs))
    private = _get(obs, "private", {}) or {}
    projected = {
        key: max(0, int(value or 0))
        for key, value in dict(_get(private, "shed", {}) or {}).items()
    }
    inventories = list(_get(private, "inventories", []) or [])
    positions = [_get(farm, "farmer", [0, 0]), *list(_get(farm, "hands", []) or [])]
    unit_actions = [action.get("farmer", ["PASS"]), *list(action.get("hands") or [])]
    tiles = list(_get(farm, "tiles", []) or [])
    access = _shed_access(len(tiles) or 10)
    for index, unit_action in enumerate(unit_actions):
        if index >= len(positions) or index >= len(inventories):
            continue
        position = positions[index]
        if not isinstance(position, (list, tuple)) or len(position) < 2:
            continue
        x, y = int(position[0]), int(position[1])
        if (x, y) not in access or not (0 <= y < len(tiles) and 0 <= x < len(tiles[y])):
            continue
        inventory = {key: max(0, int(value or 0)) for key, value in dict(inventories[index] or {}).items()}
        if unit_action and unit_action[0] == "DROP":
            deposits = inventory.items()
        elif unit_action and unit_action[0] == "PLACE" and len(unit_action) >= 2:
            item = unit_action[1]
            tile = tiles[y][x]
            structure = {"COW": "PASTURE", "SHEEP": "PASTURE", "GOOSE": "COOP"}.get(item)
            if structure and isinstance(tile, dict) and tile.get("kind") == structure and not tile.get("animal"):
                continue
            try:
                requested = int(unit_action[2]) if len(unit_action) >= 3 else 1
            except (TypeError, ValueError):
                continue
            deposits = ((item, min(max(0, requested), inventory.get(item, 0))),)
        else:
            continue
        for item, quantity in deposits:
            room = max(0, 100 - sum(projected.values()))
            amount = min(max(0, int(quantity or 0)), room)
            if amount:
                projected[item] = projected.get(item, 0) + amount
    return projected


def _public_signature(farm):
    keys = (
        "WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
        "COW", "SHEEP", "GOOSE", "PASTURE", "COOP", "WEED",
    )
    counts = {key: 0 for key in keys}
    for row in (_get(farm, "tiles", []) or []):
        for tile in row if isinstance(row, list) else [row]:
            if not isinstance(tile, dict):
                continue
            for field in ("crop", "animal", "kind"):
                value = str(tile.get(field, "")).upper()
                if value in counts:
                    counts[value] += 1
                    break
    return (
        len(_get(farm, "hands", []) or []),
        len(_get(farm, "unlocked_quadrants", []) or []),
        tuple(counts[key] for key in sorted(counts)),
    )


def _clone_distance(obs):
    farms = list(_get(obs, "farms", []) or [])
    if len(farms) < 2:
        return 10**9
    left, right = _public_signature(farms[0]), _public_signature(farms[1])
    return (
        abs(left[0] - right[0])
        + 3 * abs(left[1] - right[1])
        + sum(abs(a - b) for a, b in zip(left[2], right[2]))
    )


def _shift_state(obs, step):
    seat = _seat(obs)
    state = _SHIFT_STATE[seat]
    if step == 0 or step < int(state.get("last_step", -1)):
        state = {"last_step": step, "due_step": -1, "due": {}}
        _SHIFT_STATE[seat] = state
    state["last_step"] = step
    return state


def _repay_shift(obs, action, step):
    if not _PREEMPT_ENABLED:
        return action
    state = _shift_state(obs, step)
    if int(state.get("due_step", -1)) != step:
        if int(state.get("due_step", -1)) < step:
            state["due_step"], state["due"] = -1, {}
        return action
    due = {item: max(0, int(quantity)) for item, quantity in dict(state.get("due") or {}).items()}
    market = []
    for raw in action.get("market", []) or []:
        order = list(raw)
        if len(order) >= 3 and order[0] == "SELL" and due.get(order[1], 0) > 0:
            item = order[1]
            requested = max(0, int(order[2]))
            reduction = min(requested, due[item])
            requested -= reduction
            due[item] -= reduction
            if requested <= 0:
                continue
            order[2] = requested
        market.append(order)
    action["market"] = market
    state["due_step"], state["due"] = -1, {}
    return action


def _future_sells_at(step, horizon):
    if step + horizon >= len(_ACTIONS):
        return {}
    result = {}
    for raw in (_ACTIONS[step + horizon].get("market") or []):
        if len(raw) >= 3 and raw[0] == "SELL" and raw[1] in _PREMIUM:
            result[raw[1]] = result.get(raw[1], 0) + max(0, int(raw[2]))
    return result


def _preempt_shift(obs, action, step):
    if not _PREEMPT_ENABLED or not (_PREEMPT_START <= step < _PREEMPT_STOP):
        return action
    state = _shift_state(obs, step)
    if state.get("due") or _clone_distance(obs) > _PREEMPT_MAX_CLONE_DISTANCE:
        return action
    market = list(action.get("market") or [])
    if len(market) >= 10:
        return action
    remaining = _projected_shed(obs, action)
    for raw in market:
        if len(raw) >= 3 and raw[0] == "SELL":
            item = raw[1]
            remaining[item] = max(0, int(remaining.get(item, 0) or 0) - max(0, int(raw[2])))
    prices = _get(_get(obs, "market", {}) or {}, "prices", {}) or {}

    # Near-clone routes often expose their next premium sale through public
    # state. Prefer a three-turn horizon against public-route competition; fall back to
    # two and one turn when the longer shift is not safe.
    for horizon in (3, 2, 1):
        future = _future_sells_at(step, horizon)
        if not future:
            continue
        shifted = {}
        trial_market = list(market)
        trial_remaining = dict(remaining)
        for item in _PREMIUM:
            future_quantity = max(0, int(future.get(item, 0) or 0))
            if future_quantity < _PREEMPT_MIN_FUTURE_QUANTITY:
                continue
            base_price = float(_MARKET_PARAMS[item][0])
            current_price = float(_get(prices, item, 0) or 0)
            # At the $1 floor, SELL does not add market inventory, so moving
            # that unit earlier cannot create queue pressure on the clone.
            if current_price <= _PRICE_FLOOR:
                continue
            if current_price < base_price * _PREEMPT_MIN_PRICE_RATIO:
                continue
            target = min(
                max(0, int(trial_remaining.get(item, 0) or 0)),
                future_quantity,
                _PREEMPT_MAX_BATCH,
                max(1, int(round(future_quantity * _PREEMPT_FRACTION))),
            )
            if target <= 0 or len(trial_market) >= 10:
                continue
            trial_market.append(["SELL", item, target])
            trial_remaining[item] = max(0, int(trial_remaining.get(item, 0) or 0) - target)
            shifted[item] = target
        if shifted:
            action["market"] = trial_market[:10]
            state["due_step"] = step + horizon
            state["due"] = shifted
            return action
    return action

def _tile_at(farm, position):
    try:
        x, y = int(position[0]), int(position[1])
        return (_get(farm, "tiles", []) or [])[y][x]
    except (IndexError, TypeError, ValueError):
        return "LOCKED"


def _trace_actor_action(step, actor):
    trace = _ACTIONS[min(max(int(step), 0), len(_ACTIONS) - 1)] or {}
    if actor == "farmer":
        return list(trace.get("farmer") or ["PASS"])
    hands = trace.get("hands", []) or []
    return list(hands[actor] if actor < len(hands) else ["PASS"])


def _weed_repair_action(obs, action, step):
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
            unit_actions[index] = _trace_actor_action(step - 1, actor)
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
    base, equilibrium, scale, below_func, below_target, above_func, above_target = _MARKET_PARAMS[item]
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
    current_quote = float(_get(prices, item, _market_price(item, current_inventory)) or 0)
    later_quote = float(_market_price(item, current_inventory + quantity))
    return float(quantity) * max(0.0, current_quote - later_quote)


def _demand_per_day(obs, configuration, item):
    town = _get(obs, "town", {}) or {}
    shops = list(_get(town, "unlocked_shops", []) or [])
    turns_per_day = int(_get(configuration, "turnsPerDay", 24) or 24)
    shop_interval = max(1, int(_get(configuration, "townShopSellInterval", 4) or 4))
    demand = 0.0
    for shop in shops:
        products = _SHOP_PRODUCTS.get(shop, ())
        if item in products:
            demand += (turns_per_day / shop_interval) * (2 if len(products) == 1 else 1)
    if item != "FERTILIZER":
        center_interval = max(1, int(_get(configuration, "townCenterSellInterval", 24) or 24))
        demand += turns_per_day / center_interval
    return demand


def _order_score(obs, configuration, order):
    score = _impact_score(obs, order)
    if score <= 0 or not _is_sell(order):
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


def _terminal_liquidation(obs, action, step):
    if step < 716:
        return action
    action = _copy_action(action)
    shed = _get(_get(obs, "private", {}) or {}, "shed", {}) or {}
    planned = {item: 0 for item in _SELLABLE}
    for order in action.get("market", []):
        if _is_sell(order):
            planned[str(order[1])] += max(0, int(order[2]))
    for item in _LIQUIDATION_ORDER:
        available = max(0, int(_get(shed, item, 0) or 0))
        extra = available if step >= 718 else max(0, available - planned[item])
        if extra and len(action["market"]) < 10:
            action["market"].append(["SELL", item, extra])
    return action


def agent(obs):
    try:
        step = min(max(0, int(_get(obs, "step", 0) or 0)), len(_ACTIONS) - 1)
        action = _weed_repair_action(obs, _copy_action(_ACTIONS[step]), step)
        action = _repay_shift(obs, action, step)
        action = _rank_sell_slots(obs, action, None)
        action = _preempt_shift(obs, action, step)
        action = _terminal_liquidation(obs, action, step)
        return _align_hands(action, obs)
    except Exception:
        farm = _farm(obs, _seat(obs))
        return {
            "farmer": ["PASS"],
            "hands": [["PASS"] for _ in (_get(farm, "hands", []) or [])],
            "market": [],
        }


def _kaggle_submission_entrypoint(obs):
    return agent(obs)

# Cross-graft: preserve mechanism adaptive, replace only frozen route stable12.
import base64 as _cross_graft_base64
import copy as _cross_graft_copy
import json as _cross_graft_json
import zlib as _cross_graft_zlib

_CROSS_GRAFT_ROUTE = _cross_graft_json.loads(_cross_graft_zlib.decompress(
    _cross_graft_base64.b85decode('c-rk<O>Y}nlKd|^^B{idV{aOp(`}5}GGz4<vxd-UU}v$wV)oFxx5fPTb&F(?RT&u>nXgKsdwioQv+8}n%*e>dFaLM(AHV+g_rLvi@h@L4{`B*Q`}e<my8Ha}^W)~?d3N!izy9~X|IgRIeEs<MUw`|rzy0sm&%gh-zy5Of@xxDd_ZPE^cMsc(+46Zd`1<pYo6V<-+5GT_&zsHruV4SPxqtY0F}ogoz5Q`>_w@CDUmm{y^!V`p=XXyhe|q`Jj~`yzb@=f7kJ)MSKR!G@{dn5!FBjX*=ckt+?Elv9i29VnS6?nZeR%hmpFbVCeKCLea`%$jqlZKO^%eJb?>754oU|OiXYl;5KmB+dGp7sPkaQnxr^q|DcTbzgs1Np=aVDaB%I5CS_Cc5J_;tP8ucV-VhX?LfYVYmfUEtZX;~0Ip`26tmaOLcGMt+!&Pd!d%urk4558tyJQMgk6{QKs_gUqHgKcI&{eYtpdcUUi7XU%->|GgW>XL~f+v*S54xW_3&!@L`l)PUN3)?Y1l6uMvRhNIF&GU~^!b`uQFguBn5(+jlUY;SiCzcFu;r_m1ean(3Gv+L;be?~neLwQ^)$AR<rN@Yy#<BpSdx<$f4O^n@f`d*$AjMfM%vU`vHaB_fFM$dn6Va)cw)t;YOV43F*KD@zWhdXP<EyaO}7n^bW1#S)&(k5UkB(Ht@QhWBQ_DSkB;f3<<;r@Q}?&+65ZyujM+<*AD;TlJO&$T-2DcD(K*wZERr*QTDbH8VB&{gh)+za<dHu=%vP;dRw&I>q$-|XFy%uh1M=;zJ-?>_a(4xcUR=cwm|SvP&|y5p<o+6g?1&nwlQAl(ZOwCGIjY%F)6;`X6~7QHrQ<=5w0!4INg0ne;^D0?fDx`W08OJ2vF4Q71j(rpJEh@y#o@~Lp7PuQIJ9A=K+S&R;H%39o^ge6Z~lW=9g9Rx9@!gD(0?DZex-I6^#?^PZ$yqXs7)jRGL3oi22%OLK)?=Dw+x62nX>)MHH?_5O+Ogoc3e%cgph#|lYf$Q+~i|R-OL&BS~pSiukbY&-VQ5}`AIbFlqQrkCdog5te?rwv*(FE8Ab*awrPI~|N@M(B03_pe2>?RXk=)e$dLB@x|;06I#T+o&M2(3%K0N1m+sT8Z*_T=OSW)me`dM4iEzpc_6de4GYkm-&!DTY_&499dr*U`=ada^~j<2IAa6*PHp@s>bggHeJP+GQ1c*UEF1juf7YRzHHxl~c;TuR9IF`<N{T-zUrmgX9yo-<v*oQVS8-Zg8Y$6B(1o2A-$U*f&Hnw(XP#WeVGw-n7gB9pf92nP`5+Zlu#djFOtY*nRl%?(tuv&+(neOe}CUA^L!4ZuQaDB_>5J<8#96TnA%9gDiwri~-|^^ATh>1W`{L=Z?kar^mbPpEi$=f88Hxc0d--HSlKN)RQ+j)Cce%<FIBM70X(+xkbuAxzi^l2n$tsiVz@reQJUV0gXeTnt+k0#nO(;owz!^l^?&(+TOrG9eN4Cj7_~_Xi_t?c(gaPb;25aV|RiW`9~~b<D31W@x6F%a$3e=nlSez*ER3Ds#?;^xnoLC5c<RhX`kafKQjy#H+9InHjQ1Li&4Ft$f&j#`YW2$S>p2;eor?b-ClN@dECafnXO~wkjdx0ApkeP`0qX40RRh4pkboF0d@g`4htMM&qu0uh<gr0qBTcC3*9-czhUTNybgi5B)T$XpVOx*;UiJ&#_+BasLiJt52yTh+(F}y7C4Eg^F*QR5_iiU3e@mu0Qp(Sg1tP`L14U-@D=DEg3vwNrwXmp26$j;qcussnfm#oSi|<4I#XJ;e6QZx@HLC;I%ko9zlvz1+@j_4>j9Ix<+dsZ`vapET$rtv6M*LuhHhJC*{Rt9y2jr=&K&@{Z}XRo5m7aG0ALjw=?(D-at!cry$;H;E8T~i#~~pwz!%}~q03k>G_}qi0jWIN?mc8icoCy#LsIh101d-;+QY*=yV5ed2>2l`LKda0D|~IjMF1ARZlVei4C+y!KZf)x(_uH;Kd==2KdMH}DV@DcrU$9jN65ja5>%mo29#>=XdorAmwPsAk<S~FCYty@;LVoMl6<qrwg{Gg+B!dd!Lo(MFpiOafA{hIv=%#q@&rc8PR+wMzSj#7_Fdty?|LkrNS5%c6E&ihu&%lX$dobel;7HRA0g(FB*FM*7sBU3!%+(6tqp2n2wE^M#lO3}i=2Z(sR)qv>i)Ny@ywVsZ0*$i6D2}FqwInKd8B?3vZW2)&jQ<t*tx{sJ~pJgn%V1Px{PA?6983+OC(2ScvIbuZJRPAfjKexLG?TenhnV!R_UQEyX-DO9oxCxVmukj%F<J8Lk(b%xD$X5)!?d=oPyXRVAEVoH%PwJ;hb@|b=KfqP-+kZ`Ny0!tzBsC5LFq$<xUKpl2s`+&~8p)j&GXI-wE0-gls{ZNAggr&gbY}sP2zOE}d^nUyb0WLRzVsOWHh+qO;juUzJdl?0+7TEk4};Wxqrx_edeT7l(S3o)O&E(G=}t2SxwZIE>uN5ZG20wVy2E1ck70qX3-WeZ^R81GIrGy7^+Ptz#!gBzqN52~CtHvl5e?1x+%vKE|UX9Q#B+;xIe0<9k6}7q%F1m)N2M4pMMME0d((hXZn)m(>6?OOdgOmh3=)Go5APr|1l`T!4>d*Hz431vp8%(F}wD3mP;E_+pIkpUXO$C!mSo?qmE-vlE63WDy$C!dOeBQoFXp>)UFD#*yftsdkvb1)#xC7|lC=70Y|2+ZtMRQG@y6VLa8S<#a{c9Y_q!S;VfupvPc!yYWz^y&{%Nhmb~Vhl!F^nh$(Qv_LhCu);y5I1M~SH^g5ybLydiwIxU!%LpI#o57)!n@wPAM0<B++ADeqMK^#|FELs5paw)CMxwjI(hd@+AZqwFBO2C|nU6w0g=j|E=|2r-OpI!PvICGf0KYU)7I>Ng?DG-$X<!^qZT8r^Y59j~!ui3p@%BWWJmMjDj{i1Qy8)G#dB{5dBh--%e0sMU!Q3~p-+%bQL8Pi2j2PP(X7H9b%{+poz*IT6P}nUOU#sP*gQ^%r_Y1GA!`yW(6>atkSPxcsAVOjz{h?`^SNK|{0?|OAgu3*IMt|5I{Xu`3ijhZ}+xgZk>$gZpB{cz(N?=T5d(iYaif*v5*$WSR)fDL?b3?OPqN$-2ew8n_3}gco3zj;%{Cc*1L<a*p4&=RkRXUXvRmiZl{JJQrFj9h<ZFP6j5STQ``fcaR5Na4YlthV%Ku*)U^$!bNEiUEjjYpO*XW|t>5wyH07Q!5EjXVGk46`C`GY~W#R8DGSnG9`$MM`c&fH!HHWumf9k3y(JcSJ!~8^yGyU5W@ofs?lbM;7M`j?5S5j*gsu$z1BOC+%`fKtu=yD_x2vyb)H&EIeS?L>+^)B+F%B?qQ=84g^WM4-tW7%gH>GqbD1I_8{ec=){ox6wH+sWye0s57DOtM2|9nA+an(e}iU@WXNs>b#jI)k@RU~R|-DWc;bz^Dk<K3c5n0@+NOkd&|V-$if5i1*NLM}gN=aw2zCY*Kt(?p`cg&5k{Y<-=8$2$wiK)a9LOCe^oj9eieXvi0&~g&U^*t9jIPW}MuFJuRVyOO!rOePnX`FXlqLqIhAPxT(37Z?p(@3$G@9J+!5I~6Jj}+($HV8Kbq;%B9Fx@^xaaXa&aZR@Qljm^H!*FlO_@0DvJXVrdD<nfml9Odx(^gvK+IB|E4X}4ayO#up6RhM`A(KEC+0E@7MrKOyG+T)fB+9t6_D0iR-zA`Zi{Bp%HcuRUZIZ?!tj<{mgPCYHQClL-w2uk4|-Y8i%ENXdj_l(ZVpPHrYEo<ou)w=d0KZaiihtH11M$En&IP9NkVYv&d;V01V-B78;06I$Uv2|9)b8W%G4MPLRU?pnh!#9L;LLTuxpUkKr)ktQ(PH$8Vv`QMfa6eM_~aRixX1DQOe6^co`QXq{0`Bqbw(4P<}Q`908GC5-n%SlMWvv9B56SAwaFD9JxJ(GDs!!ChZZ;cnHoS5rBBScPmg)#Cm7NIT?im<kC%80=^q}3%`hrKTcbr(Z{0*rb-238<2|Jhs??}<pSo<wPzv%$ByH7t|QCQmD&Nwju=Y0(PoLknN96xi8YBu63$*aw}=Ur(3+%L2?h;aq^%ZdFimtm)^c9ELgwnaQfft-6f1SbtyKU<xwv?3wKvgc=Ie3zGrSOc8VVYRKBGsFnJfLOa7D6hMZD&Xi8FeW?fMd|KI+?NNmHj=@2XjnL@!UJ33*PbcS-=2?i+>VB$wdL==#i4`r+6QVB<-Xr-Ea}(OT%We{tComjlLn1i09_Yat26QtPYRgxD4(@I-qi6u61qUYZJaqXU@xeq=Nez?j?pM%-jbiBauL+X2*E4ga7y7mNzvX49})SzlE$27sk#2V2rE(W|F{*dR;4aD%ZCdhD)uatfI}dR$F`^W>1OY<55*vS8N~KS>W8bbS(r-1X}!P96w-i7+Sg2(>c40CVE;yPMMuPVpX`eTRw}YK~$~M`+omQ%O~EdCx^xG4_wl5xHO4T*x#5JAnj?jL%L93=5lTxH$dp>EWllrw3bZfcu-d;?oV2nCx{&1ZwJxGj%|ExjdV8nI5rTwFu&+IrJ8#>W3Pjs7Yg^IoypKL4?;CM3}8Tgnp5lu^U_yohjxjiHw)!6~?o^l2g$?`RN;{%FNo)r^Xb@wS)U=h}7ykAoUFaQmfa)qe{9q1vqL9>0;(&=+?OwAl9q?fhZrrpc4KBW!6Y+pE^R5SQcRL*jVR|mgX{|%LI%A!ul*4&d3m8c!=u}&yYoC2!a7n4q)Th)!)B#`2{2bN&y~&+A!=^0&1oVEdHEI?_>#4g1=Bo77H~~tgI?7!%Hb`gc_!72J%uwnHoc)VuA+vA!R7v6WBR0%N7+J?lrnvH)PGQE;lTC5D=Hrn*zCQ26j>f)%wbzDSqIOi3?l81;(ig9l~-4MKgOVa8}7w6S0*5TEPFzYNH6{okhliY^RGErCA0J<Avz5O19t_%pRQU@gjyLiX><{gB@tqtzmWk!oI9B(2U(<L%=KQiLBh4P>=(d#o7X<e)#XE4)u`D8GIt*JLH!lcdKEtIWd%gY&7wW+>BviDr`xt>h!5*6oQx6;Qj&v*e;#=TT}CQws~Ef>F})O2v|=7RH&d^=CTQ;XU~U$W!-O&jg_m}H3Cx!BXosK04`3XXT)vG@I;}bl;uXaTi{LwsAgN@n_p(cO|9}i%aYZTD?cwvyeu&8f#%{dDXVGW4Mg9TKy`MJv362&5hTIKE=?(g${snBX-FeeM(1%cuY=x~Z2GSB>o0-~>AAu1H4p)}CkPk%7U3Qx5ePuD=o9#FG`&a3Nn{j+myq_giOsKCbw2nf3sC8rT3*y6U3LwUg7rufha!ZmI&xb@v^Gt|i{f1|GM&riBxj-sUv6B&uO6GT=cUaYq{vJ6-%)*sm}boV*g<Atyu}zr1TYrkiq7hqNn}KJ8dNcl6w&xdE|^WS;)h~ca2jqmHuj(+T1J?^he`7Q0?%}rrIBPwy9CxQwsPxBYyl7jC7#95D81hjZM7pK$r@?%uN@HkEp-a+l?x_|%)E2UsGFZJfvG=9*CqI)h+hFkuJJfcyn4peisG<wL4OPErniK83dOEjl1GljG$FD>%P;}=nZ$<+$PSH9Z`IVxioe|U%yGC(l%2$kYb>zK)>JqX%<H8H8>5d=HJ%E_(!lD8W*+Dhso0swej-NBtq!>;)v<N)d@+&+hJ<y>OPjS@030mTVc>vLG{>20D@i?=_&|@``%(BBip)64^wgx=CDg%!y&3T!R$iX8R*t+RT;)>THz2qL&TjbScH?h4pHh-b3Jp&KOH?_!RE%H|<mc?M7+vlq0n!F^wv5alqILlBtZFrTFd<xsi9&$1V*nfzC=H9;cr#>>YcYZUHb^EKfHk>g&f)q1=9a|&1}I6+)QyS4a_LKfs*`J*#pA#R5IC<AEiN(3`J(5x4yO6WCAygMaZ^Jr*`i|;*91F<9I`6j!pk$Am!{Decnj|@7dO&f>2CsXp&pzT)H@%@Md!WQn46*`Ddq@-D+Al!l=2{UjY%8b1XL4ql1>8Fu*&O1`e<8_vnpaZEX_<#<8lhqiJDr?&Qw^0yriJdg6X;^zK;6=@U%pO)!I}QN&qoe#M<OaaWA47P*jeer<g9Pz0ayd3Ao5OR54AUC8|ru+N~2asCz)u@Ek>O##%FX*hD##yb?NB@k*UJi5hn-(=c$F{o3DRu{Hq~n6EOT*$1#klXxnSFBBjnL9k`;0Tr&Hq$v$w*R`k<Cm$^oD(XIoC*@##swfRCs;TS#vA`?iZX}XVMU8P0b`0>Zs?eW>S*bm)<~oX*B}a5cShh{<mm5QUZ4;PICrBNH4J!c6x>#DOs(gEX01aJ~ZKzCKQgr?rsG8mhFC&FN!N8Li0a1LPS4oPJl%>N}jM4#g)!;G8Px1UH`#~ucHt8e*E74nRm8pw-^7=e5I72P2mP_AK)A59q=aEvHkm*Lmgc5OG3UX93nHb%TX66au!;+klQDNeCAq~n?)2_eZ0XeJEe8_7@g_Ab13+**k5`ZWvU<G+$PNjPl7o|q5A{7)?UjW(+^R<@Mm?(nHPB<3Yl~RJZn<2#0NfHwif;X$IV;-Q30Z-U4KSqq~?rpo3os?Vq)zS+9C$dtl7X}<ekjtpit8mI7r=Rr3#_M7xyEwqfq*#amq|)2xxNJC4SlSm{I4?EW<z3FCv`F!cU_r!zwOl;Ot_pU?gPb<4G@GP&RwMfykPw!QCMw_Pm0&4X$|P|eA;qqdo|3+}0lSG%W#<(rl-(+dC{PGPqTWxX7QOtQHT9jc^Nmw>LMK<LOm>brQh?7w@R9T6>_8+r5$vCyx1({TG`@rk@r;)f1zGek><mC1IXx{$sFC8r4uvLnmLsikReGo=mU%r&l0z|BgIR}}JfnoRi$uzSDr8LC%0WI|CxrCn6l3`M*vAJg$w!ot;VL>5LXe=wS`EUQZnT;ViO&o{G(Oxti_6ggA;JTKt>+#A`m47@B`qb=$=fD*FN@U%vx@&5WYJJZFS>5!YK0nMOUnr#yTrw*%@P5@T2U$QU?HfK66i_DcZ=oF5j^3$rt#GY>Hvyh8)Kz$O@BK#y;!kOZc1~mS?NEmqy`n*mB4`}K#;&|oZ~{qur+y@jv!c_Ze_3#+Yb>cUM5tMOk!%;fu0!=&pF*USL2H(>(CDHw9f<GA?1^0HpW}ieJbcqp^8<uV>eTgdq&s4$>%wY3-r`NJkjYUDa`iSg_8($Gu0cSZ)hhJDa+;}Fcjm_C3^#AenKVWb!Bmw#;=_TmI&v{$mXLX%)S-`!I31k4)fqA$;(Tf6rvHTRg<RyZb`Re97+`;%!uS{>|#0-5X}2HIYF!v6BT5V0_3AW>$)o!fVxpK(Iy5Om5i2aN2YP^@I|!k!v&W$S))v`bsut|m#DCNa57miV`|bO%^i+C@Jr#;3air-U{u*7@xFnJONae9d==nfG-_|RQv>=XBHs}<6Py5=6L~t+x4Ii6#?z!MI=#ZSmVoC8s)?DI6x5?l&WdOym?Vf6N82urXeK}K)}VMcr$|xYqis|QjK%H{=kuJd`KwGdMLMW-7BZZ!sfaw+uaxhV#z>amv%AL0?$Q+R{(z}mVqU9GqiyM*X`KZ-g#W9TX~l)WYR>I?lKifO=I}-;$|NN(4#LkPVqE}5VK^9eh{sldjz(Ll=~XIr)mq={be<A^BP$qe5wO5Y;|d*a4Pj<~u3c%o*Z~$@TvoA*?w@gLcCm)r51k2Hh}I~*fpc-71vSy0WM4j!3MoQ2NeG;1Mzg~f{3m>7c}U9e6rwQFEo1`ew5A{nF0c-x<BrL#gvS?1rn*kx7cp-%!ju_!x!Zs_CIewp;LYHO7&!|_v{#4NqTX&lOHMA7MRks3`dcP##Dm}1<gy%NhIfT6zpj=1<Gj$M6#SsHnXnC&{9wZT(N;##TV3v$fg}#B4E%&aDw7ZjBQ11R@l0c%h@#z*$(knfxK+yNh13ibs)F1R5R_1%eocOlB=6J1>{o6<ueXzDW|^vgbV888sXbSOi1UF`kPsLo^Rsh8i&F5W7=V*<$|uYX6WLJA++;Ph<UEU=_`zZl22}qN07ehhn=GN)NkLL+g>dY_Gx|}FoI=J;DIP>BilHH4u3x?Oz`CZ%6!4u`5xH$bm+B?Ezd~;URl`^oq@9I|n7NpkJ&w+8h}KX{m*dlLYj8Hf7&^$ZMqJ=4XvMo|4QMaO2X_ZV2f#LF+&R~McH`eN>xtxatum4C(7lU$lgc~JmW0-7C<Q|s&}a^FBL(j`l)k0@7GCL6fItw-kcOA4QhU6$jqmy^Xl*GGCwj#;4}z<D$&HCaxxwMT%jR6`(m;&C5@)1(09mXGXtc_QL>@u)iixgaqXtg6pOMFsw4CT+LUSx+;Zb|t)_2b}`=G3Q=^!PG=RdxN&-&aepC6)!rK)1r)093x(&5I}^>LxRv#yEmLn(sxvw{S@z9vv6h^Z?Yu?YKHDFiWOie!uFb-omJpc3i>F&?K~OxHRieG^dsO9kT-QqW>L`$j^5+SEj)B0#{VW}F9*GKBKw#;Y?1j1FgAAt4=asR_bLk^^k2M`+ia#GkPp(-IORx{S8Y*5j~&f)m*D+`~D^gRJAgM}p2zMq7ts2M+P9?}Mxol2klTQe=3<BQIzyuaK%yB9T|~1R1}XdajPDi68K<!m4cB!$st|<dGEiC@So>{kZ0<y%4Ar5u;~tM^S4w#Fr9?J5W_=p9aO|q7)G>ilWte2O}?KJN2`JX8r{NzF%dsEJc~1m}UZKGExD1ng_)c?YA2hL!m?}gZvEhtEev4%rR6l{ovW{tXADPy_hdD3x=ZFK~k$<cEh&Y5=B#Nu=cuTyTQ3}HNws`k8M@VV-vjjW}cGgZM{B!3D_$I5hMGPZz094N(ItsA{w-l>M9EDNG@7xU5muDF&3poV`xcv13qch)VO?3qP|tBPF48xs3=AsE9889a!$kfT9veDn81-dH;g#IebRl~ugU_$td^eygW#ZAgnwu(AH4{bu%4*CPT*NnxJM3RMhe5K=<qUAI+*H_QX8Mr00I_5Rqe^$>t51E#Lh|PohotZrs`5*t`Uc-HYpTF=*_ZG?HcO1Qa5o+nP<#Frjh&{|NqdTQ(xLv+}a_239UlvSu;|Py#d0q^$ba5(f!+|^$TxkJ5?p~K&ivd*7C|tZl397<+ETG;Pui4ZV@-BW_K#fv*%DP^tb|JWdnj&Nb8D~-)VfIbfvJeVhPn%9f+qZSr3IQGMYc6q&pS7_NJk!<*QoDy(>7@wT~E*O8GPoQ0;WO8e$AI(m@Ga9AXv|2{jEBso4R37ibWmnt1}8N6h5JSvH+2)ocX)&yhjC`MHzUdW6WeE%<s<8SurMq%8)&8C&J*aw-Ur7pIE|TxzPx1UR3bpq*lwy%EaC;Jz^PC{(*pw6gbG@LH{YqALVr6GJI$7{L{-df5UL0ahN9l!(}N3>(W<Zz`jGV6iC$-$eS14D_wCKp)WIXG9LhPy>(B2*{lw<sH>m&zaPfjBZ{6c2_O72FFB6y3kTD(I==RtWvaeC?AfqT%3$z7P2kS#%+=kcsQq7&WusTdUWM4DLY#&2p>Wcm1S-#+C^K|RxJmA5+V13DYB@CPZ)1cs|LfXxy5mkgccsv<|mUu674i`iXkya!{n+c0yq*yUHCFTwhS^RxXS^YicF1uQf>rASWud09hRX{HD&{5$ea(5`xJn{Wl{WWu!)rdX1eLpT|H04IpNx&;^hIMfR^ublPH-Nb;U%chC%z;xY4*Y<kho0=3W5daujUlTLZ(sD|5->4c}g&K1P`hNnx2R`*FmmbTE$U6+7qo*g1#-uq7akL|BN>z(%OP*FF7kNmGgiGVGJ^dYwPR5_ZTn4q&WskONvtBEQgnz-8UJcyBa$5D*<F1xHSZCfZ*qz*H=ZR#nb|^ROK%;v}nC1km)z<@BUvM}l-p^{iEt_N172S+YJ1aqnkE6*)$Zx|%__>yJ+xTGs;vOlVm}8Q`i-X?I^YO%?N;BA!ULSbSKENo$`7#YSRTH1`e;_~wTnve{7fK4LnRD%-UW5|DhKTgRURoRL)vcB`%(f-E`}T>|r}tpr>wYe&g2G@M*W7Zh{9qB@ysrl*|oX8AAiuv6eBV(5@cy)X@Uzp4_=)5nmKfGu@YWIhGC`WB7hD}vDmMHZLEiztXZ)L~D!VK-@A^~Migd-`d{%s0H|NoCup2o!Npl&m1I9`?zuK5Fs#_gnI4NMAnd(XFszvU)megnt$+N)<Mymk=<U&0O?w?EAZq$4f8=Jdm-qkG0=KYMP`6Qns0bOVipMk>?UsXN!K7@YQXclj-2lBaMg30ZLUu5&|Z=EZ5A}s{N)^qkF!{l!#+18Ko$j|ITFuf|^Ll+n(fvMXo0`&aIywFB760#1~}{Aj+;0^8ah2i-Ol73I^6L&G>WmJ&FqcY#CD}|J>PLx&T?bg9QMfFffKnA3A&6Um{+`P*dFcGND38BI)cbQNaQC%&gUSV@ELi8#<9fV+8%qqLh7`a9%4zMpet>x^=lt5_=a+bB(#Osb)~9)%{^e#B^fi!89?zX;3VSEE~Gsz<~KNquSK~o<}O3LK8`|H!7`U*Eu_C<T}mTCRaqibhw)FUl^3F)TxlF!YwlqKu+_~)msAY*kHy+8R-5dwD&IcM`&!86R`hTF_=}tttWxyILYLyqK=h^RR~le@mfUz+MYS42V7a))I+sGbVh@eiZS~!1|E>RMXGQ+83Kldq_S4@e^rN?T<Us}ZYk-)q4?`^-6T>a%64NTR?MInl^hd-%wpv^Yi1~l`2EDC8X#9o8VcMuHLo&kxDwU0;#dg=72nXr5NTj}2ON4anK3AUZ7<qN79G1ei9)a^lI4lb$dt)&!ALdISvq1+>rd<fVZD%2I5v``A3%H|d&g5?h&L6`$keYF=mj1y+1ITIHWEyVC?@=_g{{uw9@K7Nop4>n8Y1JYZoQ-;(Fpd>vYgrIv=QuIkTgP*7@a3OnSQz<9u2P~(KORGI5}-rr3#LE@vyAioOc_jgq)k2GElcF+QjmFAV$b@O9br(h@sg{Tawt7`HEdr>los+^c1D3u7J2dPqww+p%mSNCQyrba+nf1u+xMGo+qW``%4MBi7SSkay(cidZ;y$N~UfOI>d5jl@4lZHNuJQr)yA#GIgc3K#Xc#s1Gu7DoHy*F8FS5-Xw;X8yew#DB$8F<q&V9lC=p6h*GDSMMW032?c$!Y1JaY6dW|CkY=0`HILX7WdNqXVVh7uF>^r?$E0!MR!$P*cF3i|<@5`RA9?ebl;LpejEkZaR^a$Hu}4HAPeJ)h6zg0p9ktH>SI;fjByBN^GY-Eh@{j~BoK~)kJK*`!)sfzJlyJx5{ISZ!81<qQxn^ptxZ6)jrHk~)1wnDyI7i!3rFw5^!e&7aEUKv>2AG-llo`uKx&(+458%RK845)R?PhwL5DtULC*QYZZ<0SH3oESDWF0mu?7)tqXA<pAAk<UYt+xO;ti#cmf36o!O5ZtiuX4|A+|Q`)WM#WnN>aP#XAb0m&7icE$eYFU%VxctN)#p?q?#!FXOgtZN;WR77&TNLw`S)$B}`Wjn(-jiR!#929I9;f(?nu^0J}x=%dl^)VuzZggj+2auSlc@EmoGypDO^28>34>@aJT7j+Bbz342E-P6*;6L8UaFyQfRjB<iF4QLCDjt*Q9&+ayRRulWJt)blnmm9Fq=s9-gc;-OS|OZZ<rY0z9iH_zj=KV8Kd2r>xS8OtjGGl>H8n)1P@U1Q1k#yl9dcRwz$xpb8P(aG*BAo|*UCAD^c$smgIHHXj!>W{-{>F(u0Z$a~I9dIeQoG2Nyx)jt}4hG$2@ZEm4L9vo0xVL9&0n#gV7ii-wja+V=8Qb760AX7{-QT_2bgOW+r_H{6buwhfce1~c58y4C{@9w{2k|bwgD2T&hwlvB?ld^x&(oLXJGykX2a=t=)Q7;-?;bSwn)M|;2KH^Kf#|wd-iBkgG4(+xp8oug=l=&4PwH?')
).decode("utf-8"))
_ACTIONS = _cross_graft_copy.deepcopy(_CROSS_GRAFT_ROUTE)
_trace_actor_action = _cross_graft_copy.deepcopy(_CROSS_GRAFT_ROUTE)
unit_actions = _cross_graft_copy.deepcopy(_CROSS_GRAFT_ROUTE)
_CROSS_GRAFT_MECHANISM = 'adaptive'
_CROSS_GRAFT_ROUTE_NAME = 'stable12'
_CROSS_GRAFT_ROUTE_SHA256 = '5ee61bbcd473c7bb59ec284b0992dc881eac24a1008463b6d0d4edf772d131ff'
