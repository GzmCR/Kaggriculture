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

# Cross-graft: preserve mechanism adaptive, replace only frozen route v022c.
import base64 as _cross_graft_base64
import copy as _cross_graft_copy
import json as _cross_graft_json
import zlib as _cross_graft_zlib

_CROSS_GRAFT_ROUTE = _cross_graft_json.loads(_cross_graft_zlib.decompress(
    _cross_graft_base64.b85decode('c-rk<O>Z1oa{Mnm^B^`!t;RR5)awyeGZG}t5^I4NEZ{W^80*8>H^cwk(vaQNRT&u(neVkmyYOjxTI{O#{W2pXBR~Dm#lQXKm%sh>my3V;bn*M2UcY(u^SiqbAAfqkzj(O3`1im3=fD2f=YRS9@o#_m<v;%V-=9B!`LjR&eD~w)AMV~<Twc6<dw+3x{cySY`os78{kx0HtHVEh*zaF`{`!ago3}q+T>ftJ_5JtzyN{p$`q|<8ckkc6`swAzlYjd4C*QyRwOxk~5C404+Wa5izW@0BX|q3F-0wep{PhP<|F-Uk`;^01pDy0Le)-3r-W|GqarNt$k1y#xdOPHwU-9Pd<^JgnCoPBX**yRAPk&s-Ea}2=NRA(3r^q|*?>_F|t3JeU=1j!#Df_!a+c#Z~<9F=c(@GNkJ3R1lrLNu%-UXh0xs1`Liw|#q+OC|Zolzg=<*CbLij@fld;6Zp5rr$|4_`JX9%MG1^#MKn)2EA<cZc<I>};6Nr~iK($A@?{#k1p~GPui>p<#X;lhT03eYRdLaTJcfcpQ#S7s;qET|G`PI1}zZd?+u_b+fbGx&0=*$)3hI)XS<#cIL68%U`OVlA&DI$}({IUg?Z!eB3f=r&~l0YGLe_>HGC5Vzef#$m4tBhm!-eGkW=h8)NqLTkrXm1y*_P=EHCB*zL}`;g;gY#M5Tn(*ifGIBhf}N5RXtZ{F-*e*F0l`}ZGTzj^&HUuKIuIJWAzO<Jv54$IMY91nzeTWo%dUbX5G`uAr4j-w-61@<koc{^r%VF8(jm$b!+9k4u4KZUjCXaqlf_1m^)`@m*v)*nVM%ai7HS`|C@bRX5)bz&wI+H?37d{E(D1kX9HoNwzfi&v+f^7NNu52@T_n=9)}pZ9OR039c1<>k%TRfdp~s}vErc{tnE=bCh+d&76p@m5O*O1BSPOZbW_I9h$P#W~f~uz+Xwoy4a*ntJ5H2`-vB+XJtA2_pb&>_{KF_s-ay))fOT(BE)$kY7CFkrtdO+M2O@3c`ASU$^g+d-EUi^8NqI)V=8ZTb14&{c+hk-ZMrBVZC!LkCPi?gR~jmt><~2$31O6E_DpMo@RVZ@T*2!ctpW^IKfwF;gEk!{LuLx>GVr>_BA<p)lGif=H~nsp7!ef+jsWU;GG<U0JCrX>#s%v$f9>F&{Q)pP~^21yas!IL5FQ{w`>uhZ5(2O$@>w&J8UpzrrO<#jWESrh$r)0vw=lx^)2=N-TOa<Yi0Q|pA~j148&){syJ=S@pB4_efaqP?*8}t_wWB4sYR_|zjLpmx9lF|>3P~S6n0)?)yA20_{GT}NCWe<5rNoX4)fS}MoivDKK15Ujd5<$SW&33SYylKV)a~xkwv;ZejD*Atk>-9W_*>Dy|KW9ON)k>BXsUqSf>Fi(LDhkK8!l$<G9+&rb%h-6%*PUvbAZSftlQ}Nw?il2IAIIJcLS~xtI|+1q6{O-SKON>{ZT?-83O3;R>^`ZY_3z97SUs3C%=UrxgUoW3#cIx|Vq|KxZN)26L$a#6&J`IO@8ad@wcym~0^S;Ui-4E}{;>9Fta3fB`45$+OP+EG6F-cxzxLIw&Ba_$B)W-3}YR?<{;B4CrxNAz+Nqs@zvnyex9rkIi}fW>P1QC{7+uyFC1r1at!20v$}i-kb#HfhRAQ;*<x+*%*P);$s_(%QHPToLtWvSlrWZuM-37YLpSvM(1|FJ?ONX9K1Ud!bHbt==_U=1t3mMcYtS%!ZqzF-c``d&Y%=f)8lqV;YfM?KArR;s#zGX!ybS8_Va(9yIVZ+&VeoA(}UyVmcB6Cc95<FK$lK76+mn_<k}1nZbRV{XjapMm;e))6=$8qdTQ~O%s8w4Kr_lZixUOrr;%EMS~x%kdd_o7KKmK3E&`&m04-MN@K#?}$(3H^?gCi-21hS$aO=zYu`C<(e8e0?PHi!RV1L`x*!f+L#Mx-1CBVQ5l$LRAr$Bc1B=*G9YldlAU?M`k_6GUZJy)|I!Jp%lpr}bds!8VFfi7S?fSDNcEqky(Vz6a1LH>Xlm^yeZG<fIm#b9Jd1z<XMMBZ{0&YGGAX0P}mN?<aATY#q9DNa=@yoGslhSB2nge9VbOy(r@*s+x2u?vtjI%M({@qhzQsnOsD0Mj@hO#5RBQ82!#GmM0tw?SP^_zoJel88bAlZi%8AlZ;Os}8}^a4WQtJXZE)m62B=y<8W9vQpkd*@4TlkU^@l9{?`FqqP#j6P6~>VyeYgVA{;-x6KIiu1c5EWS-?{i~Tt6T)3>B*8UH#-~90la8`j`q8+<^Rc=^jaii^y-(}P}uz#~YkSp)lV8vGev6gil-C7QthPpfpK_^4+z0}@TW)|V6o&Ic;KR{^<aaBp;r{+9ZG4(E`FhoaJtkrd-6fUKqoRu7ZcbtdqxYvx})hZ18fw~rP6grr7<^o2u84Va{^4cSxzeXTDDrU<G>5S`5QDTMA1A`47r|KlT8M7i$`|JA(ATjK=wetJAwA!TmOzV!R`K7EMHA=t%*~G+FFNADzKzQopa7DkwQU0@*hBZqx3+&h?d0|ETu>$ANsLRp_z!(a48q3EvL#VlsCumf>1fyR$kRxtWTfs4Y+xH(!Szmx6bJ(G$zgx#|xHGw9UyCThmK@`^)_S|!sM-_Ak|6Fcu#dKG<mp$NtDs331#j@p)zgqzW|0JQ`+7iIhDpv~^KG%?dgRCJ3i<IqzgObLZD*`os01)!scY*bYK<|i*&+^Js)+MQgOa*zuU`MN<#EC-kUK$X)TWRi#8@XX#^T%K*4kGVV9v-nK3A?u>T`AU4FS(<(;Axmm2B7&7|qU7613XP8Zi_{U0)fzJjEOF*vUsU#ti@sadi+_qM%t~2FO_akB8f9^ujn66cM2NXjYI{It3sm1~wnim9M2z#j!a~_<C^ygWQe1gHE(F5{b?-YFO%YG+Y}(9b?U+V9&;JZG{U$yGU@3k5TleCAs#-j+j+Mok5TmtaR2$bL_%q*;$&<RyzM#1CV|u!8D(-YY*057DF#9KlrNS1r`I1r<Xa&z8r{eL>oqB1)Na?MVK>&X%nLw4c)eDgML*Sn`p~KuH)2%wFrX(H`D_IJ1v6QCcsos!xd+9fZPYrNtz96j~JS{vc?ZO8`X>BEYHf(rTIISyaYpe^(ju_Y=GUTv|JI#OD0Mv(Iie2^hk3UPme-OM#|%(myJv%A<Dkc0v@^e&lSn3YbDZH8L|_VAXb{wl%+vb0vXVXQ9dHw=L3ftL6vg?tNi5YY|42BE0X{)_vo4jtfHEAMN_xwnTDJs%2YbpB`ZEnp4gl3r(KDff&<UYu*}2#UE4a<RZ3RY{1%|8Tc%&w961(5m{NAiHi1y_+=UuTo0H{u0v<bf{E$e%tyk&Q&Q3+GI;QK$=8i1#=My5EgURh8$*UPt^dCm&=tVL|FTfE~*(kDqP=bzEgxnQPEK-9B*h(>^lr&(f+t*4565JT|@9GazEW+sEt<gb7FM{+zWs8w4b$1>f!+2J<Sa29wB3&Vfp_tHqgpon#X~47w|5YNBUivy9Ci{u1kl`pD)hL|cS;Qd3!lX4732h@`)Fp^|!)$wNGQywnAv!(=8&MFw20mO8$Vu6^{fH-$$q<7|18?kz=#L*m3exbn5tB5MbCu|&l_japd>RIi@QvO-9FXz7@3@<#4?ngxdFHs<_&kG?Y7knp4M=o(A!S860f?U!5$#6r3z1$XHJ<COR^^ny>B|RzHD-(kO{TGsz^!H+`mjv`GfjqbQ_a;lr@`EEb1il;4QwMoF*@^HX2MBk^+*h$fwB+=#QmcPqQn!t;>S;C*e$^muBXSp2hjyI$1`clRW0Y!h4LNHcrMA~gd?U9^;d2@feCwcoq<6gKM)ENi2U0kaMgMeCs{l?&q(N_EVCoXxTk+Tzx;J<PNYuzGH-x(-B9MB3Mv+dxlvJy2tJ8)O@SN9us$>6W)xFRE2w3qN}?_9>*h4Hq-d~M`O__~r4JvnggMSlCz)HtiMG-k(rh^u>>MnSMs6K&vV&=JHPo8+Hg`I7fuKAoyE1U`a9MfDk;cn|+Nv1NM8Pp87vnIV+ogGSTAB<<&OD`+i1I`{GFjZ7R?cYw%phBurhE?9k|H(oPAJGY8Nrk3qL8N!)=Lsjbc4)P6Y8X@^>9wMGsqiv84+-*N-Pn_tK_tSoMJYTKU^9&s?f`9q=rsxKl6aeUDO3;YmWQ%nNyr+5dktsbaHyJPX<^tdBC!L9=&*jd{UqSu>50|AKvhUvIOU0PC86#alG={+hij{$b8J`jOMxW{ir|VU`l>En;Zj|`b~+^8WW1wH=70!#AFmf)()lq#WeXD`e2kzl>w#FDW6QRS3E+RJws>|%V+UJli<eInZ(-CDX1Gnqbq9w4J3@fP)vE+Pl`ntp3jm{?jm15_g)oKWP=qUvEU-|5}4aEEq3PdpyCb++{;{F<u?+iwh8Gl!UHo{6XCSVOvcNBy^z8WWjA`X3=HQ6$f&*J-j%!(HgePh;U1lT8L+{kq4^R$zUkz~U@CeA2>a5O%u66=?PE9-SnUg(lTY2(X=dD$AY;3J##U>V%nCziRyg$Bz7>y`mUc^g4YW2&Uabl*vusb4jMh-qh(0WgeSz0vuzi+_6$|Mf8xwO$GPGO9s;Rb{;Pzg(%hx~^;!Hf;W!d!^pjOsDurnChy0$n%&B>>s*Wu~N{Fd^cic?JmOUumg;u^s^<qbw_Ou^#CB`rL!kVi{xe?Op2^Y-=*;kPG7Oag6;N)CXTVrs3Qn&<Na6a_tzW9SIQ#4U>2iwGCoOEqN&hV(rk$O<xaS&j@nJgnS|G_Tz#TqO<{QGzL+rSKNA<=cq;pPi0(M0^w)9uAo!anJ`;!Fj%vr!miR4=NmiW|D*_E`kR=2XSgD^sJhZ4-dIvPEEp#0~eh&hR_C#cnJ|p+WOX0>{SaNkx-CW%$z1NCpmCfJiff1+V+r<;$92MON+1dnkKL(E6yLq%e!@_x)Jv20K;ONuIsJ-<3-n3w)ZnC-ErXEbV|53g6o~~u9kD}>u`0G-L=t8g8z1cJX4YNOQV<K{=!Lw2oC@#!MbQ<^IsQ=gxZ;#1q;B=f0$4VnTP>{v_s$r>4=6h?FlJ`JJUJr(GvS}7pu)ip&4nM;oy0!WL>JZxju)kpVArGR6<U99<87e!`{pl#;e!15|hLX5b)BftTQ4==oUWq=m0I+KS?fOi=k6=i0WE!ioE$rpOLW*l*r&J+YXHax6K*tBhm&~Iq7CXy_*hT!Yn7epc1bk5O<Jo#a8Otwuy4=Cf1!8(InD~1}nKXjt&?NvYX!7mam*mxg~pu;vAuXaT>~=8aekCfX8(|8)y6+9<3=urd&>13+pSmv8_-#mn*y}smHNI&l*Sk^h_y;!B}bDAdIhIPO29pLLmcQB{+5Az@)74f$B*zog28;Jx)A2K2kNF8ml%7ICQni++5MoB?{0Vq?%35+;LK^qHh-><E}yOx@gxrdc`UTv47*XGn7#E<Hk(a$Edxuue)v}D9i?i1<f!|!pOx}933y6*ZT^MT*tj+VNtWZZZe5^_2Tgib9~$3!~tLbxIc=t!h)c*;@EQpbSvRT&t>N2hw|dR6_8BiL`Q!Ojh2`FT=wxzBk@}M7)dz<;Ef%kkCWusEb6zx&{wO$=35WDj<F1}n~?2M2iiH%!F!C9YT!kh2~1jsmrOLMH5N86V6f8$MKv=R8Qt~w40E|@kQ|*0`*Y`VpZewNJ7FEzK9jc&7%ropHE>Hu3GUlRDwVbrETcjtBm8$&@QmNplN=#R@$%r7l2V#AKUJ+ZM??<zz@}3*U}j5q7WTqV6m*LXO`^+l<?v!Jfiant92=s$5|L0!64qVG3-pXeZ;h}h98($(C8X`O%~(l6q{5~Oj<O}y>G+OPAa)hu5_ACPQJEMEp$KlCRjUF!?HYytX=vu+Lejg({SCL)GL(?1UMuUs7|GRvkGfS!NbTOyr;c|N^(dJ<U7;v9L95n_c>RBJb*O6e_ynDi;Dzyi#$us*3q>V^0bYeNBY?w1@i31h0_t!mk0l~>o1)87*pRyY$&uklIuREZ%h^RAAFYoALc-zojnCMm-e)@qU=P%{Pj)eBq>Vj!fC`%g=2aIt=y<U>-K;9LlX~PuH99(la`m7R1KoiomDSJIv~<UC-2f1p9cY-X);^7~|6;aOz~2}7L9-wpZKtc}{W^{Xjz3ESW0?R2!_^WZVDCbE(J8xLDOR_erd$9$(Imm!cJHdnbYH8oAWQQtW^4=4y-(PWncJo+ILoi6%FeT4F)DSfsjQxEx67O-y7r{fA(f(z9v^~YYKE>D04X98Gw4hppa!vADoIALo--H;8uT>tnDJBgRvo2r?iM^R8JHpK3}zaNb4VB<b}Q45xDr3}Q$Yguq&$+uzp@T;rgW8izT>qhDKARKqYYtz61R$Ef@!Tzn&H$c?Lid)QVNclrBNb_b)^MO<w)KZwm$*fLeYKCN2blD8Rx96j2bdVIHJkqloLxkL$1+*8zV9(%=~yttMHQHOeAQhfp3BoPVE;G+Z>Q0h;U}LX#{7T+c2gp5z|`0CFLk3td$2$|GdO&`5Fk!)vi)={Tq;E99C+)F67nvWdJEPCC}}vNnq=^d-R?dG{!U~^#l5uy2tJ1i=j)Dpa%S8RqFvV;=5;*7Sffl@YJ0%Z+nu=f#s*uBoU@)*pk4Aa91A&G_x*vty>RR1NLDJ4;=YRF!=f$Vuigr`jobdTL{>b3ihms2FnE3j&4o2U3jaGzc+zA|J^7H<J7lPf^b>x$yikx=vpYppT#jc`ApQmrtyA1$rJa9NuEtnX3Yq1aIF3O%N+Er4YM~Yu3T)Zs0Gz{V3}X_0;PjX(mOWrkgA1P$+@EyQ#~J+GP+v1|7^f$VhEG;=YQWQrU#m|uhbZDkc3i$A_#Nd20^#g4v0I|38`o(!<<r_!4U8G_Q)7lyRVSWV*L(*x@{~VmwI{p#ci^bZkySo3e@lt?NR*DOgSP14ah;@K?e$fbc~O9d*YiW`>gB>=NW$V>?Y6kVqvNx8ex}RQ`Vz2Dm!Qn&#;I>hb;6@sM<s2Ogy4H8&RZoK|{E+Wa3Dw3S+52Kd7b1KHhZ6bK*K;M5NCK>UNd(PT8K+=MX6#XA0kwy%_f!AxUh3;nLwix$T0UG;#s-ccpMMsVY7<_$kLj`@fz*gk4qFEV*6ab3$mA3f($oYLk<%036WtG?CILv35o58B)owsHb3F8&T0ALJ=K|gZku@Y0+sC1EN9bmMOjhNyEtTBcc?UE6TY6x5l}nQdAfdB>oY7NK?eje;>%V029V49eSLbxK_dZvL=A+3)Wo}{TU$zu)G>n86$RXUI+5B0_R{j<Psgv&#tVQ2gc8wZ!-P|11To!>>HYBa##wLDEG-?Q+;cwfzZ+BrIX6#D40{?$$*&fONA4{>dD_b#A#-cS94^FnGcT%qZMICA(0?VbXzlIlp^9}d9o59!iww3nWp6@I-~_b=ka;@T<ye932HDgt|~wttLumkI{|gDT>2S#PD%nL^BEE;Lvapls-{}<0F7Sm9tJdvX)YJzhUA0Alpi<^t%hX6bAeSOXbX4%g@j(t19%xeoP{!vE6^+b7cO_966Gt%YL@pvF&AAak%Xp2+8?Zhv}8RdJRE~aQCk|6NT&f9V00<A(I{$_@?Hvn3@z7Q)zoS6J6si8HLlE2?|{OwoTIoi{UXOaE$G;4M>AW%nfy(R1`#a)Mv-VO2Hv;`eNlM}cC_;mO)#<+sg#WoKtTX$16DaD!_0K7GC@_95Y*~A8}~Q{=@P=&5W#;YG(FMiiOnYpCqUT;sTG+_4@U}jS7hXT4Ru)wT{Trw&d1W>_zzFgS#gnGgbj8}G08#{xIo%r`XkMpCLf(*ektmMP&uHsi7)e7R*H0a11v<$I1)8?HTk;I(L(<?LBgL0QzJ!J<!#5qy^2FdQUkCtXep#N|AT?M&HA~P)PbB#k!9ZQ9o$SH0*_~i=&nTL70pUj8QrTc#Hayo9VC?l)kk^>`Lf`wU{6Sr0{5l{imh*OMi|a4X#r$L3<8Y`L=2QJMtg=5cpJclQ>nL~YwO<eMiLd&hr{rP8<v}(w+@_auVOT`moPq23jO)1=$=e(226o`YR+Kd1z!2m6dQ~uM$>FLRC=@$a#Ua|g%ivUR@h}i!)Gt0kpz@P^Tj&Ia>Cy6S})6a$vil&JyoKp;ggmppI+K!%vs4Jt5y&Lmx=@@v$#6Q)lx~l*=Vs;w+2&=9)pa0n4N|MCQBkSWbaGh$*)<p)3wyhvx-UUuuKaJN@V1_(>c7IQAx^4+{7!5Z9Plpc~H=2cP6^q_1Q5)Bt2fQG4;(W{bc;Mgv@4_*JfL-C70Yv?V}rhOla7qq@{^+a%g>?P_GL{4%iS;h#^X{{!=A|k*%x|V%M%RNAZmj5Q6h0CXebB_FZ*_gV$5#=4OjkQci36mp1$p=MOVo^4zgb<Ug{l-~v&5g7PS`X)ejD30%}yVpoN*XujaI>OL^`K#>uWB4`$6F2X$7LF<?3)*{|&T^%h-nA0reZ)76Hb4M}V=N$nWG2(3Oebw7$5>VniQm%YLyE7rUvG$r*krLgCgbRX|!DaCnxqe~OM?5Yz+~N)zlEF7g;+TdaJL*XV(x{hFjHHP@A$TwYH{}_?3mEp&*TD1Q{7vNVUXWl-xR~OwTabDRX-vdCUFCP8-kz&>&l_t-3iOa2kKJZ47vfQ8@(O8NSe>i;{UCTK1-Cs-_9_bJX<io=%{L$cwfoK(yRXA8J6~>O4#ESCsBm3Z36cyUp!tlEkdcB)wLKmQ9ya@}6hdxR+rF-_k<|St{g#A#;mboe#KhHbHp_q4L`^C_l5}JDRnh4gYCTM}+zR*iIG9-ebU&!e1&z9`J5??x1^XljBSZoD*j*5Y0h;chyzpMdM6Y(O40MS<S{jwUrgj5-y(*PZudO4oEl>YBfu|$Z2iE4KUb6g4v2kUVs#8j-i6L+{A?I^pk^0#E$6dtj72ls0s&Vn05ox9;&fGekt0aXkUJLvPaNJlja#Rofc;5JmxAgU#0mPL24ug5$Or|0lw~DZ{IvGMbj@n}`65*Z|tRpcIpL(1oW||LuV~jf_Sl5u{M2+F$U9SpOAYn<-S`D3=4}DNrlTULMX`yjLw7XuUj<lJ}T(dT-62}BN=?v+(=Cy}*jAi3}dxf)JR2Vi>^{u}OI+8#k88PbSp&(V%Qi<|V6JJEwuCK+CeONBn{U8vstLO>|Wevw~*!RVEubXL?ivB}O4Q{)6){wJp)HB`%lROy8u>>pQa6Px{v%AKOjipe3H^K+H&QDH9%CD=EYsCwX@kW_-01Fh7S;q*=E3OE5NueTN);>PAL@_E1qN|T$33d#CKU|@RtEI%*qQ@iGk*dWGfT?F=pi<AOdjC#Z{Lr>e!U+J<B*hrpBTra#Jh<&*RLmze)Mrw8A7-SPRnBRG4g;tv+LP|laPN_ZsN}tNeLY>3cqq^eKrgKa;X)ljoEr_>GnFtJaxa5z`4XzT_|Z+EfG{blfUBuP`dqdCv}d_c_>v{OWs52zxf^W78175=#`G=p0IbGR->kR2UY$<I?`(SCmF9=4>U&X+ONieSDX{KxD`!D{__C{(Tl)6w=}&f|&)GzhCgwVcDsGpd=qp})-?c<Kgqo*U?4iyWFb-zm{*@4rcW>@~T;7PksfzcsY*dq78e69_6NE)INQN;|z%sF>&=c3J_S1E!_({n-v*m87MY*UQgwD?Zv@93ma6y89imjqLtri)`Q@aS~Ol(TL(1l7#1fG+<V}jw(hFM=~MU`$rS|(9>(3S2ko2Cck`d1L7h|ndC;p$^|B%3qaoS3HvmTJDm0NRx9{O#JhEFTDPo+9kZom!O@z-Qa!SDA@3)Q)y|%U7i)Qy)M2=B(sp(@?molu~t!8K`y<4+%&k%m1vLbYq;IKmlc6s`xIjnqsXoZ9$L*r}R$y?5DXmY??dmUaIrdRuowdB14>u$4j$<FwVr20o%pBTcr*%B=&R2ElYnniNv=9#|$8dh(5C%gjvq+M$__j_#v!|@ii*U#n(x3#pDMRS;P?sYBiPPq~0NJWPP!Y2-t3MgCf^Rb~###Tqg}BZ^cy!KrrKq^0Lf0m5nVPYFbe0TkOze_%5DA=g~tyM+H9#^ZDXP{b%}`M&HroG(Ta&>R$KX{PKNv=|4E6JulBB^xIX73}V_eh5&^)J+`@-%-3nMJnNhF)ytfuOuD*5Wn-+nsF*@jlQJc+dety*QOC;!Mu;kc?j4!s>j&S(pwh~0QF|+6j*ndtU#etT`q`kfGHXvgl!BNyF&hC;XMo@}vKJ|)C?W$|YDaTlk}=B?bPL&HLR_U=e+{8Gd0j(li_%B+CCd8>$fA&(1*5t}OA;YtB^WN2WRl#dwnefEU4?I{+8kZ1`#H2hTCtFErwwGaQjbFEaCz;__mQgZ&`VE^0o8ja0PkW1luzm_55UxqEbH*z)U6Wgd@Qx9UvjpSqU{yfi^v6+tz4(wm00Xdj3}Zn1U0J`3XF}HNCcH60&ofvA_$?FIM&CX9eJxoeQ2NIwKJo`-+=nh)w61$xn;5Vd0aHYLP=5_C3&2NHT@}aR*P9|zWkA*cY&NYfEV~rsazZMM#cxL>g7GDhQnn`%E5S6{7rQOFbygt3fmGFnX&B^)$kJ`Xae1A9UiZWiR$jzOV!!QOde;^D%mt%EJi8KmLpQL*obUJ5P5Cse$!gF(1CML4toK390E=lC}5gS1crq&)H2qdp5l^QA;iu$wH8sSG6A(76iq3Kyq5@j6pEKsYD_7ap=z-W0X514VlV+epRB-FP3<*Npe_kyDerRBU1y#bh1Dw*G%9gBh9XK*i_q8ITl!J}V$cbaWMth^e=xBVpzT?g6Fy-FA49?t;sA;98<7pX&IUXNTsQ@(Sf(p4`v}LF7G6u2g{(H<$f$z_l!ze#_6Yroz#lD@<~YhxZ#cp@@f%X#BP)}|8Mz{D5(3oZO-s`g_WD+M%Kkz85I{6>6b@i9>-zM!*BSP5ZKrL$sD|BHBCT-3T$5YfW`tyIys%PonA(NJ(jB0%F$uzw@axbNP=AWL-?Xva>9I9~nZSL;*z#vd6e20tWN4LgqxBhO&{k$r6sYP2p=jzId-YVC$#+T=PZin5y(qMb(mP92qr}H4?9*)_cNmoW18wGZ<X<W~6>63{!h|F2<O(N>QWcPhEW3lYmcxrmkz7ws0}9`W6UR8~V#AJor|kk7CR81BYlV}vu~k&)fKfejOD@HPk{B(G*BOYY0+M!Cyr?b8c;tGS5iX(;8ao$-L0RYEK9}9Qwk1LHmano&Lz`pt*w*shE={&$ub&lKT=!u}cfjSWo15;3%D|Jsc0Odgs8p}{cwn|gSV#v4KP<!pL*eHzF;}IYJp3QSe5Pd')
).decode("utf-8"))
_ACTIONS = _cross_graft_copy.deepcopy(_CROSS_GRAFT_ROUTE)
_trace_actor_action = _cross_graft_copy.deepcopy(_CROSS_GRAFT_ROUTE)
unit_actions = _cross_graft_copy.deepcopy(_CROSS_GRAFT_ROUTE)
_CROSS_GRAFT_MECHANISM = 'adaptive'
_CROSS_GRAFT_ROUTE_NAME = 'v022c'
_CROSS_GRAFT_ROUTE_SHA256 = 'c234e990fd63a168535b55de1f11289fa2bbc563b390390b49d0d65169cedb18'
