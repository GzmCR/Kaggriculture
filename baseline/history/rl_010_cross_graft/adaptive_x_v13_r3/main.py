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

# Cross-graft: preserve mechanism adaptive, replace only frozen route v13_r3.
import base64 as _cross_graft_base64
import copy as _cross_graft_copy
import json as _cross_graft_json
import zlib as _cross_graft_zlib

_CROSS_GRAFT_ROUTE = _cross_graft_json.loads(_cross_graft_zlib.decompress(
    _cross_graft_base64.b85decode('c-rk<O>Z2@k^L_`^Pv79Mft{&+LmCBC{WZkyo1JI0NXII@E&IOw%Gq}jYw8iSG;)fA~LH*jeKj6-Bpp1QCacv;>Az@clP&Re*Nd)em(ocPiG&lKYlzroS*&Um;e6j|9t+#=a2vV<=6lE+y8$4{L|UncXzwb|J6SH@aZo<U%!9%<Mqwi`Ps*}yWNMg^R@ZM>)ZY0&mVWYH=qBwf4jTBKRbUp`}2>xo7?wi=d1PM@c-vWQonos=T9FdR~zL2>1@CIc>hJ7_qTWVZ@+wcT;$|;Q}G^taJ=x}g!piG`{vW@`%ye2#t)y~-Msnv^VRP^ebK>0it*-5jN!uL_oi~pSABE+diS_!{buH$<PMLzn_POnM0gALOXOBWcf$^TUhw--|HmqP)WyR_HtO%`J`eWx#U`%rcX!8k{NrynIhE@1+bMO9*Bux6bc5H|kIH-eQYYn&iyH1Ue8-x8xB|N;Kv&ivW<TS*baVqzd)6RgH9lQ0slLGy8q`NkZLkF0)aKU}wKiHp7iHlGb-v(8Yx8%KsI|$TbhVjVb<!4CgRc?uugSwzP!>>#uOs1sBug<LI;qHhaFo_f?wPK-$$k9c^p|}+OB@Fe`Z*id-5S1-x}Ncy9uLr_Ys`<<uO&x8zvdcGF4ga1F}v&bjp-rB>)V@~-Rt|G|G2xme|PilKaXEtl`DR{{nWlq{l$87cl%-4r|IMF=C{yoBJvo)En*Pi3AAdw-m`h)nBvQpld;=gHvuti(wfv9Lt%G$Rv?ZX=Q};U%;>D^*PEYjN7q9;U_30S((&PNG_^W}0m>){@PDmO*KlvE)X@pEO6|JrCjG}sNF0v26hW+n%&kd4SK9kt%LZY~ce-wHk}R}vHzMkE@3|8ImpgoT`1W$Q{ti~nU*t+GyqFHit$&{?D1`RU_0D~-|1Di@=HG5J{_R%vZ@Htp#nm*$vr>v;j~7$2j?94qx0v5vh?G*UYVwwC>N=_-)x7<AmbAC7Pyoc-%Gv)ax3o&MD*`o1c+ggzcyh<W5;Jcy_FAvskmxi_!S_hJiT7)X3O5~J+KCrgLLiep`3f>TJEefa=6A0XaOnP9DZQ#%&r*bMx-huZW#w9-=O;UF|1KW%g$F$A<3UdYwB9~6#c?g<L=Q-pCMTLeof?(`yy`fGxSaOvB1f2z;vglG;|x01kR=z~K`Cw#mb+OjK|cKN?e*P%sE+VP$do=h|9t5>sAdok-UG$6bK|b!4z2j3EDEH}s%H9o957?VAh`?jrOagpbxBd4kPK%^n(u$6-a7ti`UzY*5}Ks45sVO5vIJxnfnYw}Z@TGrCGhFX>;OR&dIdW9*|S<adIGE>$32U67kc1kVZb6DAsxg2awah>09<*#rtp-8)%xU`sr5VC8LvE@k$rl_TsZ66`0Wfh&uTKK=0PbqOoc>Z>Q(S?k(6L)i&A2SQG**hr{qk7A%&lKNxzg}wyYHjmehGgfo?xUD6X3mYd9E_3-&TlEk|=TWd2@$(h}NVw1@Wa*Oz@m_f7wZ{bX@^w-T3qMJyVW_fZgAP(?pXZls`&U^J3B>5K@<1&UqK895l1-NDl>JKpV0qH<!F&5}s^8M4V0ft80blNC{bbBWN!kHU%qDp;c?^j6k?v9gl1d%X5snNo?*8)`{OU8Epl`vv7z)RGc5sff8ZauJ5Kx3@Q68syJ?rwhE8RB!9%`u&@HZ+{%8&D-~}dxN|K(JOp4tMh!kzq{W5u)Dka%h~x=`~s$3?|!joxhl;ZbTl5fKBGah_uq?J@%6@&Fmn$@)8m!F|DFUH3LdlXEUj&?$=uIlg>M<A_u=C6LL4eOOl-V$4?uSddNub&$pU``Xadwo22Css8%K&h5`-D^d<C7P6ayEJE#vrP3k<0RifxQ;*vH|+yj~g<T^f9Fd22yK5;_8fT>6%PdJ|h}wT2Gj>_#SpV=0i)*T&ieG7I~9LBJfv+Kao(t<5+IS%JS!-l?$K86PAab%*Re)Mo-sl@L_{AF@cTt1$f#ccNq+k(lS*lIFec)`X!vKS>Td4MO7_XVb`Jn+TnzM>l+#iyUJOsrS{3`0X&VX4wvYEO_#h;PGD_#G<Hkr(q4i_iU5w)j<yi(47Xc>9;D-x6IZWX0-YHU^jPKfNWiaBpV9?8N07mn!r^N<)ZunEN(<(V4ijXE^eU>=bmPWJc<-87lmGAU&cI|Wmyha;7)c5aaAC@j`fZ4OI=h7^3>*e2ar4mQRVQPh3!Pzxeh^<V9yCO(Z^ecd1BRLB<MQlxuI}SA0XabJelC$NDH5f0M1o|z}YPU{UH7j=*j-Ct)Rr&^$@nUB7sd+cxS8GA&AhF=sPQ7!T2!sob@Wl$O@uJ&n1gm+KrhkVODug3dXPNS|+^t!?m;E_I~##QSy)PZvOlmCP)OX=Cl+s1Cd&DIxY<>e|Du$ukcMNO63II(<t?2K1yAyQR=EGN`3fH18rFngqGlRe!0|pZ^kc-1E*XpTC4!h`ig>VeLqP^xxf}Ro~m?OFzggQ#w*pf3b;yo5JUwoz>V#;k&X1LEq=UF09xBvCe{yV^pa!XN~)3GnumvEQ!CySun4<_VLehT7Uz8%t?HQBV7;cm3aOzLGPi;ONIL?*a;P%#QEbeLleD9~8u#=&QdO&(3H`2=6wrA^$Z?mWe2E>3GF#GCbl$HfW|Uq+5EuSrGAvLdtJev?AOg3GlnK9mH9bSEG?VL}Hv~tlFrmb28}!EuJ#@oK=yfKBvrtG}VFcBSg}|E3WgMiS6~_i~6dn#%K<tZTl_(@^ylel$QNiV%vTpmJ5xY||1hNX<<M3L$SC+WJuT3n`o&=NIz}CegUr*#HQXudc(cb!07!8m^;44wV=xLS0&u#j#hw+WK0V@2yWS<E{qPY3YK0)%PFQgOfz7K}P553`_=nViA^$2ujQy$hnmP1X3wQqv=7#Aa1Wt%qnu|0|cJOh?XnhSZUw@OnCs{c?3pq1GNE^n;5sm7>W<EQah!lH&nERn@0N@sF<lY|Y9!%)woU=L^z*pj@5b2gfy_UwhD%3AKvgP5#7_@KmT+3cZ^w#hueVN+?kZ0QZUrB5iiS5&H8vM~9umdy*N2>Y^Bp42)Kqq)H#S%mPkyu}ax7oBrer0}%Zx(-KSZ3DdRR7c|j49){T@(Uq_bqKLQ-{JN8`59OCqGt<dk@7;}@U&;fZ!KgIsI(xo{>q93Dg_G*2HK<>!-BH(LUi+rfVQCG9k-1ISR}A6zvrwEr8~l2Yi(|bo{|mb4QjQHZX{2<iR@$8NY-GiBbaH-L2Bxw&P?MR7+l=7M|5iHvr{M9wGjJly(D@<)!K?PNew@LMV{#SxN5m>J^oMz?i2pczIhRl(u`5#Dy)|3aG{1z0RB`_61qWCsC9V2`ze%8LA*xScwX#&WzhzX%5{iNUp?p~qzv36f!V>cu0vfi46HBR<-a^s;q%dhA&jRF0S*-4taT?esA9eX$K6&R1_?f)L#>E@<78RJAZ+hKXr%&SFh#zMYl^b61WewrW?p59Hl_~fL9fGIkStR3Q+X}mHmKrDA%`LwwLNCz^2bcr1)orc3_8)Cf_QYO+(Z_(F>D+wP&yk(_X%a=&fC7!&anVU2&DRRw%S?n6fa!rL);6qECqs_H23;!Rxi2`C8XO-E{e%piySJE&aJkUrg<r&)f=K@lU}13sLfk$)2U`8?0XXwM14?1pJXiC919RCK$6y#or%(sE}25-oe8~1MzPQWrpfG_wW9#5(voYsDDw~Gys=pG4XM;iJ*Bi)aPeAy6Pebc=e%kY?g?(C6Sd)1^2o<h9mzRB(1^f9mQWhwBN)ygBVvx=fj?iBT@1nsG+rk&Ld$w{V7q9(Y_#=a8@tiHH!w<^Ko-uNOT~?fPzinO**oRV!NXFe5FiW%WDgY1Y70!rgFDn5WVZK7RtDuT2*6Dy;7#`J6LrmSK`>bz4-Kn~uSv&u^sVy+>vaW?OQO=!NS^7{UIC(kZRge#skvkw_k8e~E<G3ovQb<XPtrtdMgh*fP!lX(vt`EC`zt#D;t)!byVqY3lK#zWfNDQ;Dl6KD{nyquy!3QU_ga@calopxKxzG&{K<tRG;=^vL6TsH%$O8%Tgs=slgSAretHHaYFs|+3P+{|^1?s?^KUhCUHSbPA){d70HF8_{=oSKkHX~1b6FJYpc$ofX@6(%BW=KMrSDzc{hbONZ_$N9@q)?W^1|q_1ETSKjM+~~AcMumaJ!YI9y*c<#)us?5?zq8_0(jl(N5QKzFz_&Ta|j{1VmS*qLfhO>BxV!j{E7`=*^~a(%fTTY!0K_{@$m-o%YJkf`LATqPWtLQR=!j{nP7I?^E_+=zXfwj*aHD?}1xy87_?4VEux@yhzyzy9-=?5=@x*@&toum@2x$m}vt8b3J3LXc{?UU%}S%jBO%;Z)NNiyd=?30=ztASfWuzlbwb=8<!bIeO9|)4qypXu~!K?w41WFRC3Q|jyiUeL>vh=E^;No`zP!^)`uBIMHP<p7yZ&BY?QF0vUi<GW7#cbB{hB7tzP_8H<!QHg2Hl@1+m7EN4P<Gn+p*rE3C5gCKC3o^@<aT*;>|+#<2!avBj$&Y9WB-i~H<2y(>#<vrHaC0rsw>s3~6B;(LmM3jwImZSnDo{fjakI(bh}yJ4#WUbYmCbuaK_RY{wJI+lVwI-=ky2MX4w&j<9uv^2u&)vQAoNxDQ)h4=kR?SJDx2VhQu>>d%~kn;yT57Mxprtu>i30TLZ)oUnPgA1wlbiw4&0;|TIEx@)4g%(x|!`Quyh+3T1Y$>`m$fJ+G62mjAx`pI&OABw`)tQPHAW1vDz!9ZeGf`MSe)1w{aih&ef-HIGKC<!jy=rTLvReAhwv}yrd+1V9T!&p`OVe%B(D&#S_$0a+?hKYU{my2&ilt3L3@3^|9;kCQlkmg9RGI{*Lp`HfDUNJ4$kbTj6|iH6J#>yxBG2&6dC)ifdHIqmr5L7piMTT=Q{Jn=>Nzv9003UTE{KC%5cM5f!}MpOL1g4*tyTUsUuqfdNKr0xE=J;?C~e)ofen9?XqE#_M*$x3c<15YTN~%d#N}GTteWg1xxw6=fCM_%$uogb58e}~7yU9lb%uoqb}ME(Ed>;>obaK^rm$2BV(fC<1v$4N2xz*8=2y81Qp&c(9B@F&+UQ(L$_|biF{T#dGbdGH)kf*jLv^`rgbfA_GOIfQlucm9kyg@8c2PWh`nJ1mrz-jSIabN}C&)v$OeNfjFXQrZ@1m%A3uou(PM%k^rqu!!aOPgox@n(kw+nA`hci+9>Itgb!2k^{PO6l%wHFp0h7-ldovPPlc#XqhoTyv^&s^yV!xDrOZ*~onhiTX1juuhCE0vX9Qua>AjrY-366r>AE8J6^LX=@1h;}em_OwD~j^u?Evy$DB1C4eOqSri<6x2s!(5TtaAb#XWl*oB2Qe!mG9cx|ic$aNRF`Ek11?x*0w#jr3R>7A89><aI(m*JWe;|4;RK3L=v=D1sa*+BR2VG7l*Iac<^+N1|mGAh;ON_kEJanugq4sVrHH{l)4#<W=y|z+~LQ7~b>Q1kAX)3#V3+Tlu0OsXB&jn%S@+{|E0XMEmmW$ABjiqrZcP*8BTL&VS8ZDENI8A9qOYhweyK{%Z#xtG8b_PM@OevWS-=(P>iiKQgmbU~P;Cy(@eWjEf3$^wdi$TRkh0?-zZ%<UK=It1uJS|67UdgcUG#%Dk2p&unVJLYMw?T8c8NuX?<((8>Jjd8f7T*+LKaA=NnWgMV3vh{E^Eu?fv|@J#bZ`TPUu_w6Mko+0I)MvAO^3~%BDIS?yoyaa;Z#CBsb5L5e@{XgiO;8iw$+DCGmVCbpn|NdHcNB!k{WX{wPKpkgeo(&xX06@m&2;I(7?`k0$ey@qfMNuCmt?}zBwCq<MF2&5iooFheXNnp=LGHOvN^qw$SOXBxdr)5{;Bk;~I_?0m1nH#xS&E7|1WE+<MI$mq2K1Ic)*c!g{@0WFJ=&54-H2Rl&%vLi5T(83<@H*zj3`v<c*J3`qw46c}BQ&5AT(9y~@xqGsBz-qe*W%@M*C>L^p==4^_H2uiV*!qmv<>r61^*Zy#vmsekjeOGI9r|SUUask;z5RQV&2$iSRz-yCDWO%<0IF8Znsh!65r0@X-zpQ8yWKs1HDSlq4I{oR%T_SsWKBBMTSB|i>NM?|t3K2vu(3{gsR0g&<CDA^mb_O7im8>F2Qq93m+_(38DmeLsK{^0`MW8NdFnX>WG;k1a7SHJPrzVhtSDG7F5C_75TE;+U<ZOZJkioz40NR?A1hK>NMkJMyO<Y`d1#GM?X)pe?VgK}P_ytP9^@$S&J~V+h6I`kY2MP``SCsN5tAaYpz!1{5E`nHs7KU{*45kVWAq0EERNOv&F_o2AE4JVY2mK5}_PArVK5DDZ$<9zByd4C2+Mw^n4<EW>N?2QqH8R#Utnfnay9_biCJfH&5sgwt&`~?SgihEcksLNKVn<#l$2;s4o;;f2$Wl&rB|>KwOmj5^xI(wi2KUQxibHC900QEChu%-(U{0iW&~SJv6Q0Nw8JeXPo)9ABdXxwg1Z`nK&kO-JNYnGw(b@tVh~Co@?a}e?;x~6~I3k;svRACxU0!*jpWX(gZGgzL2$`*+izP5$=E?~%p^@7e8`%w2HEd=mshvM&K-)g!s*<uniH!oG2ZTD}=Zuv+S#E=Wa&cj`o>R}!moPC=21oifU1WmC1I>@e8<WIi@vcazsi$>sY^Eu<5oE4kM{W?Z6oL`jXBc7aYg-c3hB>W6G>-f3*o5$>@b<;?1K_N`Ism8KD721oez``Rj`KKZN0$bK5&qO|7>CU`z&Ko!(jg>SU^oY$QxuscsM&O$&H*eS1ORA)woDm7zLzxvRh_gf+l1x`xY=Ohh^sV83Mz3ZL-#<<hG2@F$f1G%pRkE^AB7BGfzU(L=ofTTC7?iAguq^}p$FfnSu=Wy9r960(Vl<n4*pJ2TVi7TnD#TO>&s7I(9;6j&4J^!Ih0v4XbT=*N3>!k+6h;AanPp+Hjuk;Jdh@T*vp|G-4xu+fW{4)zLcJdu%SqTL3dn&(^6oyZ_znmSPXGYq=o5SbTS1@z!+xbv$PzNyT!=WcQ?21kEt-cpwNS02!$34ErD-h`?dM(PW&gk$2QLbm$z9R8c<jf8ZJDKL6?p;{zfW>K5p$Ab<yZM<|JkHUM}`<<yO`(^G0Vf18P_Nmg*j9Y_OXST~ju-<CIOm6+G4EdFFjq351D{tzr{idO@TvI;DjKZd(68L8@!>)gV-~Ylx^NkD*GKQ6Ef|)MrFJ)?ABTNUc&G;iIIiz#1L@mUY0uee(<EUCqviNqOiDZP=#YK~&=)*&Lb%*Ct!-Xt|z-tu^Ra22Vss+cHAXxY)-jR~s-z7l*c4xsYNChL4nMhD?*`W2G+zrW|d4t;`Ut;*VxV&enI8uv2Lt)r3jByMFG2Zy;RF)gdt~hBN0xt%>xLi)IT^ebqU?>g3T71R?TW>+mmvVAIY!36>!hWQ6Q%iDW%!7fzhvjL{6{e=<cpG}TxdW(+M!C8J``f_=!k^m%%=9X75pW;yC`_+<?JK~bP2`~>z3E75vcwk1>740)I2{#5uQCAXq2MX1czx?3Ynhp8}%0^HNic|uS{DK@tU-4yYuK*K+8fBc7rbs$y>E_+t74KaG$#nBo{6XG|KAU!~?j`Dd&P_2h7?`DI^tQ#|V7bnT&J$DKv4WU)e)HP+{0HqR5&iKiuYzJHuT+24W1ruTFoqA2?K**6pFhHA}xk+Im*Ak1^vy9Fqh*sbo*xU)$FKH2^^VmvoEd<kxotwb9W}QTG7|GCE)CAc>DFfrYC6uRi%7bMB7lpAzHWB&=+YX+hH}C%Bd9VuC32F04nRbO{*6eVCxQ#5M!u#Z)1f}U`Wqns(cu(8Sp0)%85iF;~lWdtG$edbFtN^Q$wuqinhnc>g_9R?OfZa<74ze;)3WhC(_rJ+aGaW*G%GH@NoW1-g)}lgSp(0fngI_Jmo(up^6_}>Q!u8-wCcIgZXpA}wpC9d$qgJmZ3`_&|i}8R(fbuNcR?6UjWB7CghG~%zXK4JEEJCe;GJ5{LiUChAaqhz-25L`>tWu-NmSd0VHu;N841x)Rt^1#gR`}DJPa8WaLzL|%O2ePl&w^Kpgy?K4Oc!`l+fq`Xk`&V_?PgRkOl^cpeEcTJ&1h|eY%E!pak&FG6q(n>&@!~OPvA#ho*gPR4oybC5DS(HKZyZRGV_x*!FZn=c+CDsadql?S8Z@DmL!cLH;H{*Fw}Ju#wUcl(`RhI^E0@4I;Ey*U|LTfdoYEjjc9jSM%(Mbsf-BAu|qkOHz*o>^yo!$`^r26V1q1<h%}x<nN*UOsU##5$3gti{4^j8p;EKeH6c=>Rj_0sKTNdBfi_|<hoTGSUWUD(K8}IWh12?zfoQz2Mf}V9<j}|jb!bYmfwMwC#Z&xI_bJQv=4du4A5{gF@^sz_1ZL=gvE7jxftdMce#97jpa>RQ0;2`kTjeudE$M5W`uf?yd@&q9<cPI9NlJkgp0L93%P{Ro!xp?LcAYsUE2o3FjW>Qv0aNh&eH5cTGkE#+S|2wj08IJlq2^x5>P)&8Je8EzOZU1h2c!b|IGaw%1E~~}*~VE?tidpks1nL`HDr}!5P@)s*)|!Mw>5>!5eOzeoE+cA3+Q7Tv>=O30kCrTV5+FiC&70tF^7|0#h^$?V{tdJkrJ5Y8o7m27Zh*PU`#r>D`hkWGF$r&fi;9Io0+(7Ca95cCnB$6V!bD&GEOA1vn{uVC~8UI?sZg|h1nN|CFxnwIAH*x4QPOy@$=Cp>T5tPUL3h;Ww8by!6<ii#7i&;j|NdOnn-R|DM$_`PDTbTfcX4k0D`8)m<pOyI#hU>)rMbj7C~+a$+(t+h(KJLg?KHlf4tcg2B!1!fgcypU#|{5HzjiF8RwcjZ&F{}WqCd!Fm)YK#l3G!mem8aXII8%5bk{zw!L!B)<mBrZo2YdtuYADLCh?H3j2K}e|Q$CH}LD5B)eLLSfg)>%PPGR-@90fxk`e1)z`5M(u;NSF_;%8jY&XJmNF}rmz3U$l-@}noyHFO5=j*wT}L;_LNDUH59hd+F<z`okR_+IM62G&7!ts#L;bb_@QLq>G%r&GIEg8i29B|aElA*qBYVQ4RENaBD>0Qaj6E}epv>Tu`hG6q6qbrZT0+_z8;o$uvHml3P$J(a7d;5ZNXk^zII5)aiPCY$5}R2QEKQ%o1Azr(hRx8Y$4{_bF6J}sen?&O3=l-;pLzI}mi&}aY|LsFl?MGRf%{A~LK~wNL<C}yF2WKJ;EmIzJJ>l^QLd9|AEea44`&}z`w-BxQGuOlg_`E)2(%RDErb;bV7OrXAe%D@a*z2;nK}={PB7Ft{sj;^$)#oH0<NZTr{nBIYktkbsiqmXS72Zurk-!iv|W8C1oLTHsxp;$M^F_6B4zStkWs61F0GK;7}}^ibVn+EhK(Fn#9>NPyHmjvBRg6%VIUvN5eaSLGRRec=U_%cGh(UWsA()A46%56E9i|%5G$8xlHti>K@Pc42mnk%z*~x{D?tJBgbo42<{vkWc0o|`<+lAGxf&ML226EQH?6HFw$nS)5I5rQ;3Zr}K~~^R11a2So@gyA6OR9`_G(f6Qw%D{6hiGY;6o#`HpFp7y?9Xoms5L&xS2qIXrepF7*Vc0Cc7RPb7q8S4kb(B#c>ssxT5SbbWHE1GGIb!F(iIGs>Z*FUH-s>NYF5|EQ?H%G-JFfTNo)%5^PKj&!eMbwl=ul)@cB;eRvRMW)CkyI!3oq?cqDdizq%L3wpex&TKODJi;x_VF#Tdeagb^=`%=P>Z=^L6T_988NbfE4?yIJsm;nhJXc1F)!NIOCDCEu2MR2ok!f!6>kQ#a%D8BDN$G6SK(ouGl$6m<br}mWsMXcjfj<|X_0(s`nNF22V*O;4V<|)3F16=bT4M(sLYc6{_Q;m`Eh%UdDgINK9q@H?Hp}BH-+;AYhzj>u5y4_r$8-LAUm9i9%4P`-<$|@}FvgP867FG=>=R^tpOoc~?|ekiT`~xy9MsDc<b|U{gn|&VMzNPDilb~GH3dEbqL!&aYQ>KYO@Nf<b?w^}Z63ow+K@+@ayP|>3^ICAdyzJ{i=jZ&H8&2ClXGGe5|2L3Vtp~Ho5m;#4gp${&!MBI$SBN+WnonquM(awK6h@PWhDpXb0)(Pke9`%GFvpzdf?b<U>b^zW}vh`M`PAvUMP@pG@b<;6sOIVlU8FBL}-24Y+3H<zF0Z8x^PsyZOWq9{YcDQVg#fHmQ?M2729Ab8LFBfv)ag=B5@spiAfLhy?>J1%yDwBn$Fo;9Nnc*1<f-TNn%7qYSV)Ozpe3i9}3<&ISIa7HH?r;dYog1KYv|dBb<Acl0!E7n(%du1kb(1)(d-9JnuVuaaXY!wr1$ljq+wVy$<WTL<tYid)=>9XC4Gvq%7!`<SB;(7G)!dLJC)sC<VWlR7VMesrz(XR6NyCs+<g4*Y(pmC~CFDZ~gk{@jm<umrnXe')
).decode("utf-8"))
_ACTIONS = _cross_graft_copy.deepcopy(_CROSS_GRAFT_ROUTE)
_trace_actor_action = _cross_graft_copy.deepcopy(_CROSS_GRAFT_ROUTE)
unit_actions = _cross_graft_copy.deepcopy(_CROSS_GRAFT_ROUTE)
_CROSS_GRAFT_MECHANISM = 'adaptive'
_CROSS_GRAFT_ROUTE_NAME = 'v13_r3'
_CROSS_GRAFT_ROUTE_SHA256 = '9d46e73be2ab901ca6fd0384221b8f42e2407cefe40cde01ab3130371b217462'
