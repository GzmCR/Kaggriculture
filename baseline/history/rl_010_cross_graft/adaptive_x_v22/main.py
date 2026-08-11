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

# Cross-graft: preserve mechanism adaptive, replace only frozen route v22.
import base64 as _cross_graft_base64
import copy as _cross_graft_copy
import json as _cross_graft_json
import zlib as _cross_graft_zlib

_CROSS_GRAFT_ROUTE = _cross_graft_json.loads(_cross_graft_zlib.decompress(
    _cross_graft_base64.b85decode('c-rk<U2hyoa{MoR<^$)06zMmvG-nCN6$MJV!FfR}7VsGcjPt|VZ^r$1Yei0XPiJIgWLEVowYs+fIn!O0Rb8DK85#N0|DFBEFTei%Z@-@X%TH$?Za;oJdptk;&tLxg-~Z>!AHID2`!B!#*Wdp4%jciY-oAU-efh8U;fK$E{rUF&yB}}w&d$$1zTNFUoSm=DKVIMOCx8C9+r9bn$Nk&g?WeQzSF=C=xVyW5e|ElFKR*8B{AkqgUjO;?hso83@&9zT-+lb_bv*CyA3nYP`ssO+liy8;_w<9~iT^f;4-fZmKEM7n4$lnZhtKcs-u(RK>YqM;wZUW*<IUMHh6|71n~q~X>$}_6yXQ&kH#7evcX+nl<kIsg!dtjsBDW&88&>ekgx`<$KQ`gBEuI##(SC>fJngryd*b%f?&0{3fBJ1tPDl0p?UXsk>yDE=+~D=?v+<t4)X8|`q=q{U-?3XioPk{vU@N;JW<TSrbaVsJdUiv^W;|RkX})16G?<TWwP7ddTWx+_(Q4yP=wdAVpw1^8S#ADK60J7*lWsP1t4`JebMP%<{xx~H3dRBk@ogkLkYp<6Lnjm25025gjeBORZsR`waQe$WpCyii2mPFl>uwESNL|nPP0t5t(>3Nt>+dy=f_=?34ldOnVluni_J!#&j@S2hce~f0e*V+$;nTajcmH<$@~T|%<NfFMW$HhyHxKt8mVKH&?(TjI-6lgGBe+F2M0f(N8n5?ao;YUs^3KWZ+g>*TF>P`;sTf0HbvY^!M~?HAo?d2j*7fVn&$pxNp%pM57BuPja5$D)J%$0wI1u3fTA!}r-qxt26K0Ltb=pn#kByKx9CHvsY=q3MNkCUx`(D!qVas<qZ*Y<<G;uc~>U8h969A_>e0ciya<~2tX3byZN-VsX4alwkm?kKM_Am9$eXsv5U2W#yZZrPvR`qYWqr1h~G{v)0l48#nQ&30dK!ID#Z!bhjDOWXl%QkZzWsz#${v0LkZ7UQ2F}HH|zsfDG5$%jXP7)rpRVSX@u`tEVn~c5I>o+7e4I%g*u$y?lmZ)&k@ui)3ktGIX(39U`3@^?oAhG$~+XNiC|1?Uktk#Ph!Z)25T<WrNuF&(7owt7%5Bkz0p7rsdrvX}TADZ&G7ILBorc09(O{h){QvqIe3?VLOJv+&fCZsq>32>ZI#~M&_!5x(17Gb)Z#Q^f*e{XLe{-Qd<8v!YOcK-R)byCeRJa`Wj&(1e@m3L@|KaNGAv{}<ke~$xZ%orqhLB5o^%wR4l$`inFW~BN0XY#G%pQfL{rK3R;C>zlTaVAS(W)TYJ;eOLaw=;oHUuFdeqtHvx$<Lm(;?Wae4LPn^Y`f3{HxmOp;+fJh{4cj9h6#Wx&*v1L(lA?}oHO11&T_^xPb0F2SIng|uZ>^MfN@rXoSFxv;4l?{#MG<c=_09tp(RQUGmIMC*f}-MG#OI(i3j?n24>S*;lPqMuNcs+hX{x3Vqy&kV{*bC0@ZRfS3~CS<tHtn{bhS-4}X2tH+0|hkJwKZr*}(n*;mA(L3tkou?bb|!^Vva)RBxvkdw}gpj@EXm7S5JVc8Wt-L&J?-Xt<7cG?U?($A1hE(xqWj2To!A<iX27e5Ls4p8A5HL<s{{)-)JGF;1`4)%gdgs#VR&E~MSCM&mZ9XPqRVi!Buh%{U^%iKM=3}f8;`@63d^Ekx4n*6=ue%;-^f0Hlvj{~=PJwNt$kh>tdi8r&v&&N*>xBDM<4-bDmJHL+K$g~XJ&;Bfzwwbe##`D*gI7o^CcwsEQ-gq=-?x9$EyfXOTVIV^WWcHn<t?gBv`@mNC+F^DdE<Vr9p)AB?kGHM?D0e~c>b|sD;J^S&fTYP_iG_LO$k0avVVikAgHEFqBN<OV<M?C?Hc|^W+ibdF9fuF|1Zt3lX>iKrr3DpA=m-?R^(|rfCYID%4IRYUg-i^NVIrfOg{=w3EUfD#VRsahG43j(Povi>1mlUp0tcSFRHx<}M-(3=9j%9UerVBo%+P79gpg>mcSXKe&wXdd)<i|CnWIToLY*P0wbJj#sFiZ)bs`SiRf(y_h0vza$~F;BO^<W<f)|Cz8u0nmMf}DXrL}CCKOT7UlMn@5KZ==C=T5^MfIr$cvYSUeKtOjIhE2a!;m2i`*2t&L-w?aGBLl|PWsqcHL2zT&)s7}`U&L`ykpUJ<BE>Mzx_~^l(1LUCGo&X)hL$r&7ulCHk8m0C;SRXdK856~iF>o-?lxvQb+IWxv&{pK7*7uI%8@|}tB$mj9da+hs!u#6pN*z5D}`YiAC9o_oJR+pn69V*&XkstEMN<=Y%FCF7s#y|Ay7C)!J^v#wWXoBCLdztRz@HvDC~0ud7YBzXJuG0g!Jz!)mwJW;4q506jC+k!IC#-V+o<>xs({cuDdhg&7ZCf!>1qL-TfIbdEhzSZ$Kohrfk=ok)rUFq$0@x9gF4Dh0ibaO(|&QqTbV>^_7#4T;+q=wHnN>nu6J<4>baq6;x=6Vds}iy&Gp*#W=pog{;No;jBg}0Q&olgk&0QKID+#(_&^PXEI)?wp75K(qkn$-~x5oE*n`$KT+dHB?Z#94aH;ofTA%uRIcEY^wvCzB#T<{vA~Y7OBkjy#XNJ~<Ixh9nFZF95iAEA?m|@6;^S$m_uA|)qGxcIdl(zxbAnH543b5JuZ7(dZnDr>ON|ivw}_+es-Z8jdJ#M(EmP;UY|>!qH3;JOFE@r6v2VMa;72brwMfI9Y7c*(R%w>;&&G&KG|jReN|d)jf4tNKJ`8ZLGm)R=UgGp6NR2GS4rPw!FbbN^Y#5HB1j4eDeW|mOh=q-k?O!+zxx7-=*&i%oS4!qwR=I+l&uiDpN=Nuvk0lDEAW#nWx|l)iNmWG#1fC>XTVIT$f#V2#C^|5DTBQhkn|}IbeBo_a3%@Vz&jbaM#b6X<V~9^*3aMNWgH9N503c3XSMH(sK(WjtJfyk~QWX+GpDeEgWW^U7CqKZ)kpLfoVNvrjUg>4qnvMw?Vvy9Gec;1QJ~jal10h7?VTH%d(quzB)bl7T6eWlv{FH06#CtY=sKLIH7j&+YBUx`+0r!+eQ{I&J=23z#_vc|MEawL%aQIWAawe>Z?FP|ExLK0E+p|{k*?OA`Csw^FB}KIo$*99I0Gkl0m)9P`|Dt2CN>`qyc-LXmu5Cd2o!SUN>OzDV@*3+rW<hAgD?;=UTjm9@iL;1$p|E*c%;VQQvPe~$5So8wMuO;r<t77b(uHBcTY4h;_KMiPAfF!BvjzMoFfYFtu8;UUr8Cm{J0zsZS_lW3V8=F+o8BZ#GAtykeAa2yG$0`v`_X2mq7SrK?pnM$-RiSbCt9_TfNotPdP3FeoipHvk8_bHdfr%D&Wevel&c4bQfUJ?0lZZJea6gnmC8$PxKJf3Ae^ck41GgWvUhmE`|wJq5ML9PJTI2zvS<Ux={f<ZuN`z*9Y!6J==9)O*NHS4t=L!Y^j|$z3H;H6;q`>i5F{w6TC0y}FvWZZj=QZs3=;i9Cvp+{##mZrL)dzWaF+^NS(G_5&MAUy6$1No&H$~^%)k>g(OT6xLX0rx0XLmi9#>)&UkW)E(rDTdd&?g)nlWZ@qii0%$X<Fb;<l%9VOwtElmJ31UZ`$(expkjBMXp>0QjHVtDOg)>d2b<kkFP`M2F!}7IV%(^P~Eci!N#n=^~cPB6w>VMkO`8)wWW(FCkjJsZ1918i#>u&E+DXDk5QDk|15`lPCIsv21ZHk*GjjT31#kf+Jl;ht4Y#dXG%!p@nplSvk9p0!~W{)^u4GLCASyvE>_})GHlQ()81{@+s0-MhCr8^4CoHq^6m+)=dX}a;Ab}!w1k1s*}~2CW22S&8bZACoHjyxkrb|WMrCuS?_Mlv@+iw;eW+a`r&=xGYkNpr9Wn?rZ`d=RKngZBH4Va#yP?UA!<ncQXYwrsE{XnsF)=2`8R#H1Qk}`FRP?>0I}AL6vPy1NXgK3On9oa=1E6{Yz?W+(!3rU;LVPYZ7%3bM|YcibhnC|(e#ij$Od&;dP$Si8YM>eB2lpL&1M~2@2{)?2urA#?_PgDNcJ~x0c!ost5LB&EcUj0!-J-4w%1DXiT_n?VoKB5<WDX`p;?BC4kXEa$n<w1x23w@J6Q=q($QzkqQ>d70(7*s089)Fz_gxAa$9~WN2qdGH~=UegFkS@W`&-a=ki!61F$_lU|R0|<w0m+fYwyO?X%FMGmoSxgBwm4Mx#>^NJD5{jBx=o0g8eq^UEQ*9yXJe&x%zj3E@E)43Z@Gbc5xCpT-_fTP@3V<E6B0c~DJ~T*w-ARBKzub^L8&Y|}t%?(#3zfzmCz?^9S$YjkJPM4!Y{oH@xDbzQ^ly$bn0X(Uz+KyBKwz@64n@ZDRcAET06Km9OIdM;N>O`?%hE0#_jcbF-<(xPcW1an>EUD4E71pp`3bB#Aef-uWuGlC$!A*p!vm{Go_<+bsOB!(fCr@Fc#%W*bQ{Hkh$Xg8J0TJ1giwIbCy^rc8FEVl-4B*sep>in(iZJl!dIR70Tp(qZF0JX?ufbdxWPC~f>vEZU-7+^}3tx^AoIgaI$9b(!e&%uM^ZO)A#pjuTaPUH|;Qz$1IW^1NJ>SP;`*%nX8sJRISVfR^adROL!WRS%rlRK&hNuje-<ba4{LziVIgP`qsq7JpjCrBr;SpgqQitoDToU*2*<x3s2Q=Xnu@RU<6q|ls!LNd+P@Ekfzr}&AuzG<FxE{wJ`9oRyWK$k9k#EvBJh_GT8TBejpVqrL+3@=emNUe<<m(3>Hg)2xiznaQCu(4&7g*|3A2|x&JV@NZUnu8@3qO#jXPxQ&kDAH_}P-K@jK`xjF4{wjDaK*%ErGelZDTrI3F|f?HZ4f+bAX_%Kwkj6ft=2RbdHV9YwV=i={bt)LB))y@lC@lig>OqMiPKp4sA=F~=+ljG=}d3>oy~GdQHx4gZz(XKtuSrQ?Gce1h_*>2KsDP(mKs1ecBl#=v%?xXry&{7@XdJ?IQ)6}tSwbcra(np%GFWzIrM=L-XxHemrp(7s29W@$Gc(rGjT&?#!G6!Y-J7b>xs&j>AcFvoC}!vC#tl!ZxH(5B&z!`<&QBYk5?Z4y|vPh^r@~PXjSx!fQ7jh2nhzU13ZB)7Ty!6rwKD<e?~Y77F1?SLj|I-3;<E^DQ1(}*t#5WK~N6zjHYX7ewAyQr7TM<TL(rNb{O6y&qX>g)@(SL`n5)4T#C>(zLHtSHgd~|2p>TrRuYN2r`GpT=^uzxmo5q1c1oA8Ut+qPe}XXdGAVH<u8i}{y|bg{?VD{VJVCh7oK_2@!kJq|6R3UC-Y#v<?KefKu_s7#2i-rk_^6WC*4|6Br%x0YcS>WE9z2fRaw3HZJaVO84UZt~+_P(F!AvVEcQk22%ZO60AY-~~y~+yKO1{?#rouhcDf=1XKeW2CvZfW?bu?a_A*<{=sqZov3wq6~)((B{tJv`fB1wK|h2zh)DAF^uT8%zlXCkzk4`Os8F=w`s<|6FHn0><Y$xOXRlrr*`>QMPDX2xOj4M~~tX{#M8>2lQx`h^ojpb|V|!IRq3AvwZCns^=y(+li+3l`V2_O;yaQM;yC7ATz!o?hQmTzv)h?g^Oc<r35d`RQ_d>s;bD&SjR9*KIYgaSC`XrHNa6M3-tnlaV|{zoM!4F4f(+{c+<q*J52n=M*&SM@*q0%Bftay}3j<2vjM_E0l_jEL3%Db__~VDpDQ3b9|y4H?QjeX>B<&^U@=I8watPNAO^>!~yeI-F#)IX@DSwoUyzl!;AhH19<U80sDu^n4xWHvmy?SL7-VAN_lDD5xFa`Smq(nJp<WKZBljZD|9#<n;+DAdeW}#nl$N><5)7uZ9dT#7m=ELg5NQ#78tsE>uqY&kXuw3EX(^+w4c=|p*DA!AaT;A=D47EEoIN)MoTmu6aA#rl-zj6E<hA-Q<+Fv(JH@lTC?@wp(>T}2en<fJ(m!+U=5sza`Nz@wsEF_#&$8)<oLjwerP^*(%gw-D`W}%O5MRf`4e%8{9+Ie%x8<r1KHxV2DE@=x9n3+;&1xOY#DK;`3u~uSxyFdXD-w+ov}8Kxz(UB1k)u>2h(JeSiw5b3jIuPmL|wXY6OMz_7VQ%B2k^1eX}5%>ffbUNoC%8lW3ikRmD+pep&h!jR5UCO^JZc8GNe*IW_6$NoY3=nc<>nyoD1q5LXL1bZU8A$FNvWa9%OaBH!Sbr4_m|aqw!J==5zNc0Ce~j|aMPei!kuF8JE^OnFsrVoqO3k+uT<D<jA$&car}Ktqa~TY<p&Cr!<cybL>{xA$%K1K_!6?XU%jaz=pSe0}`3ebUt&?*%JL`vFwiWV@vZtM)at>=ReaD8c$GsfUDf(M25#3NN}mq|j@+0Up;<Z@XA?Bg+h8IMCBKDP2^C0LhTOZKFePS6!J&t1H}-y;-Kht1LQ&w&wc8V3*Hf;l(&tD#M48qs$d0>~B>tNBK=d#2rKuS5N}9X{Lcy$x%e9QfPqPr>?9r6T6EoxyWI*0w5@^G_SYOt0Q!bCB@6BP%r`N#zHZ#K42Xy3$vRnmObYJtNSk9inoak=Wz=nAQF1d$CuFA(2`ncLvSB?MV#-mQv@d|4wcnz?b@P%?O=+iv62P0bQb<tPU|0bod6^_E*$FpBn~1)Srjd^q{>N(qN~xe+OZQIkz9!%5m2EOU@R)piemYLNV*jZ8ajIYSF}RMzl-17wYI0|tsK<L*d5?bvD~jyFC=6_MigI!NWw07<x#=qM2RK<5@Ibdx41moJbSGS9blPN!?QDmnmcCBLhz%YffK-}8fQw7j1C7sH&KtuUeDo|I9?$TmHaw|BW(?-UZEb)WQpDZXH7!KwXB#^STu`Ac@H!J-DF@aqme{bgYe&MX=zbeR!05w)l%~n+s8h=FsGOHiX0Cy^hFN_+vQ0ZZ0p~h!B(!)TZb8jWEEV;t~#`wQe%Y)f9lp!$HE<4N`R}wV>O`#TI>J~R8ba&Zkx_McEAn@c^}$9TY7~cN5*ag)tpK%_1dxrf{GBe;wFs{NhJzq#4>1Exq){ifP?>^*c0ipC7HEEn%9SrOllL0(I}ftOdJTlLHPsv_C}AjBOxqV-UA#o4RWcgMXc2uv9{$WFe*(h%hb@a)+nBg9zh}&l#)~?T|yhgmN*$j9D7*gq#wtW*3XDASu>kv0e~aTpJVtugjIV|HGnqJ1i)sO#71+Vvy-A~Bq1xLd$|p$e%#&Nzeh~SbVU?A;8``ht?&r-oT4~a5LXKQNn2h9U%&Qi^R~VCPwxN1v2Rv(-PY-45^A&~r~tq&YWm;xu2U{QDha}w{QxwP*nXk}a<epbj^&o{aEF;Fjtjw};-zY$Zf!Ah%Sq8L$LZRfgwwTonefjl!E*7LORPInFPhe6Cv-^Am-hejVrg8o9*&qL&*MsLqdvSYsVYwKRnsi4R2jl!7vu`e(Q$TJ{S#a_KW*Q2+xb*2k0GLU&h=}}YDHO_$EGgS$s$t9UZP=cjlP!QG|@iQOv2ZLKvdb}242(UV_V~EDDwtmNviimIzjc>rLTkv3#}Qotn!`aK*j5Lo!F<+I;tQ`yt+PI!WR(E=K3+w=!tV~qQX@A$z{}q%))BV?uKr`@hK8Ncr_ZGcHT*p5GgMsfUu=T2|*Eh;tXe{O#$g6juufra;XhD+)1hg9D5e*L)Jw#&D+~yy%$q=qz;FlywD#Mg}K5{U_YG_&6nXZG`%@PF|X6IW-{Uke}oel-m0^?3B?MlUaD1klP{eaMc$}TMK1N|NfbBjeA1dpy5N}|e<eYXTbFP-xHqIXmP<~PB^5*!`cSUy`O=#UtTaD+_#NWdMF;fo@+2r>XPGk*@NAbq)5u|Ebg&8e1Da<P5b^=8Y~^1QQ_L3s1)Zeo6^c#Ufrz4qJcPD!<~j_9l4cAg$SS#)*kOSKV(}VmUs4*GVH#*mF~ksz{ind8XC+i}=*oy=bPKYFQs(4&Qz&=$mD^DTZV(W4)x?G3D&KhB!DIC1-M@OCBo~(fbJJrTQp~ICN!v9}C?rPd<_xJ*?F+57QLNj)@(MhxKYRWXd`JyYsPJT2t7~su0W~JrB4Y5>rHwf=|6z&3nc)M;=UElyqa9#~&4rf*=E5;f(8z+R_>2$c8n1*<b5u~YXZVz;Ma76=r`1je%&ZDC)sp?XDD(@eGyw_)76b^O;nS$SCEe<#g{j7r+9D!+-+p_sS9v%d!x#BTfHXc2-(oETp1;UqT9$3<cf-V!+M$)zT2<R(e6!`0x4Lx=B;Cd!=&*JFbJ<GUTJuhYCso(7#ZYP5*7`+gQjsv9b(HEt%W9j-rlPmJ&y^a`?m)_3?!&@@I1p9I(mE4Tku<K_+pMcadi^n$53NiT6u_5P)=ITflvz7u2M1MWX{B#jHB0N2zR$H*r_G!hA5fj|s*PsGfYvCsleGASjk<2)CMB9wBbuY6z>!o)s}bt7ng#Y?rn&**)n%x-((_#<(dDGIoMICc0zNzTVyg#fzs7rzg%gp*bL^N3{HbP)ov_zpn{3Dqn~CD!Kr4pMj9DOYsID$DWI<<Q43y+&ke0L{zpS$cJ;nvEu=NU<0wl6lW~-w}(T*!HXy~8XG`)DHAS$aRWtutFQm*n<VbmPXKtT;Id^bsg`^qAfO5l`diO|lI(qsDwQck3_=6~YhK@2vBbq3KkJN}(rR+4*^gxCxtslCr`F6RL1%LACy8bR!;5bou5UtRw-CwvX>P}9+;10Z?z`&EQ)7`rt;Hw9ExIvW})RC6(DLvzeBIZVPA)5U_zvBCf_XmOo9D@-ZbuMvY-JljAI6S9XjnwYd9g25#kBV~N(RxFv5CrqHq+{!+CgXV_=7B`8^<T|N@Y_%l+PKrH)q$LFfT!bYxU<P{f9Zp?cyiCJp(sg>NGB_Bs-QOW6h_z%QTU_5Js1kE0lE`9#!h=zn5S3UONIZ%wp}aIW$`aKHI$_S%AGpw9ZkOf!YY(50o2;1t11)i4MyP~tC-Tb7j@qk~(Op#S(?dm^9z1ars?I?lxhzY)*JEYoAR(F^Itm%7g}0{GtE$3Eq`U?mRY=_L;+~7mZ4=S>C<%;l)nax|*d~0&v8h3Il+Pxso`T;NSZr72)|TOZlVgRXf?Oxop>4m|PoG`nLtA2Yfh}g|Aq&p$eWg7160Z##V@6wp25q{6Qrk6#n;dq~g3P$fmxEB9UbCz>`R$t&DqDqot8Yq&EIkt+L|967OrpEh=dldija5W4sv8&lNzieY+Exq_mEN6HnoDoJw<RxtlE4^meMlF`(obT@iOp@J^W0dqWgI_-CE0R8Xei#NHig^@<S)K1QomJ^<dsHu()w_b(vzU?N7jV#xtu->T}8zRkqyZZhEn!r^{v$RbI?}Ujy9B(q&3DEY8ipLR96p1bxJ^sa=wKSrX&=srZ2_{5!vg9@vyYb#lBVQT!<S=3se)E97uN%VXIukO6__`>I0fc0Gf=)*l7W{jAdg<x+pE_+sA#VQcpR{ywE<I{28W2CbaIPhwxk2d55ybG;LW%Q7WuMpshy-JRJGq+l9_Hq*HBTidb0cDbNAo+e}ww0Z?+LU8LIR#5TOFwE>VjBmh&$jUo@>>9vDit-7LeHwKh@g*yv?x{pSD?)v4l4(Lx5Tc1QGtSl$rkrxKJR2du&Dx7u3<O&$cP?_GL(p0J0Y}6DZj#paVU&}W3))-^YgQ%X>HeALU3zFAtlTc(W6=XGasDz;!hslCEvxK;INh%wjES89o3#ou`O32wuNv9<=RGz0|2QVxela%HH+->VYFdr5j2g-p`Wpcz|sfm2kU^x;+3CaRceejTd%}dz<G&t%mTLXBQ#rFiGYFq(IrRCJfu#M|3r4gNDTltSGIkKs3HN&@w*=ewOfIrc_&c?2X*<5QP>R`H0ky@x|GwWPYCAr01JiP%dVV@`C)3E*M`XY`40=FU|y1lRR_F#k`vUT=JWqaF!(^7OY{UX~{!nQUp-_}(ZvNep50cTGDBJ7qYV;UX`$k-JsLRq8~6%X3^rKI5MLU(q`u60y#fMG82(F@*A<B}g<l#5MX6v$5@VVjjwH0h#pTdb(xQSR&y!AZ;GI?c6?oe5>hF)oTxDJ4EMy>kMwd!<QOeT;=j*4k>+KbQ}mm9%H%vQC9MvBEm4dX*96m+BO)G_^xjq1t3(ooE9blOtLYDgq__0p~ckXL)?)Z(yyhrlJN{<kMKw@yvl<phq>%vRI>$wXpj)4AEp@LlXTcwXWnWQ;{7R)xwoF1TqdP+6vI+=n$cdi0r1=OS;ezqDZ#_Z|qS+dXQNo3{7nbrWNUSFf^J(0E4Q9rJ0=!FhRLENyPrR&OV!SV?RO}R1-C|0GopmQJ<sIb4Z+4h!1cpDMCe5B|}r^*ZUag;6-zVDS<pyO?Zi6oKxnLJNU7-072m*m>$QcXh%MSHylefSvJg7)0T|@F@vW>XsAxR0)!|2o3f`}u=EySSlG@t;BmTaf?w>WO|;-*TC#@zB7zhaaFjrYwze64TcpNWkV<Go;P{56SvM>C?kllLrp-sVA#;{RMM%X?t5Tc$@NK8cB_{&ajCt|Rh=bZAQOGrTGTd05h7t(e>aBX2rccNorthpCSP92%%Wpl_QQoEuiQCL8?IgZbkg5_{$(&x!U&M-E3qmT8M53(<quegI1<6C&WeSr~H?Fh*pi0Mu9e!HO8+I;Bcp1Al%O&efQ=Y&+L`I`Fwr~CVHV66mkH`N9*sTax')
).decode("utf-8"))
_ACTIONS = _cross_graft_copy.deepcopy(_CROSS_GRAFT_ROUTE)
_trace_actor_action = _cross_graft_copy.deepcopy(_CROSS_GRAFT_ROUTE)
unit_actions = _cross_graft_copy.deepcopy(_CROSS_GRAFT_ROUTE)
_CROSS_GRAFT_MECHANISM = 'adaptive'
_CROSS_GRAFT_ROUTE_NAME = 'v22'
_CROSS_GRAFT_ROUTE_SHA256 = 'cd4380d55c4a13c2ed4fd0c9463268c5764599f7e3f58b91e960b49d7dfd5d77'
