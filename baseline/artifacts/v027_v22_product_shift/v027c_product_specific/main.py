"""v22 price-impact route agent for Kaggriculture.

The production route is a real fit-only trajectory.  Runtime adaptation is
limited to identity-free weed recovery and in-place ranking of existing SELL
slots by the official 1.32.4 price-impact curve.  Ordinary turns never create,
delete, or resize a market order.
"""
import base64
import copy
import json
import math
import zlib


_ACTIONS = json.loads(zlib.decompress(base64.b85decode(
    (
    'c-rk<U2hyoa{MoR<^$)06zMmvG-nCN6$MJV!FfR}7VsGcjPt|VZ^r$1Yei0XPiJIgWLEVowYs+fIn!O0Rb8DK85#N0|DFBEFTei%'
    'Z@-@X%TH$?Za;oJdptk;&tLxg-~Z>!AHID2`!B!#*Wdp4%jciY-oAU-efh8U;fK$E{rUF&yB}}w&d$$1zTNFUoSm=DKVIMOCx8C9'
    '+r9bn$Nk&g?WeQzSF=C=xVyW5e|ElFKR*8B{AkqgUjO;?hso83@&9zT-+lb_bv*CyA3nYP`ssO+liy8;_w<9~iT^f;4-'
    'fZmKEM7n4$lnZhtKcs-'
    'u(RK>YqM;wZUW*<IUMHh6|71n~q~X>$}_6yXQ&kH#7evcX+nl<kIsg!dtjsBDW&88&>ekgx`<$KQ`gBEuI##(SC>fJngryd*b%f?'
    '&0{3fBJ1tPDl0p?UXsk>yDE=+~D=?v+<t4)X8|`q=q{U-'
    '?3XioPk{vU@N;JW<TSrbaVsJdUiv^W;|RkX})16G?<TWwP7ddTWx+_(Q4yP=wdAVpw1^8S#ADK60J7*lWsP1t4`JebMP%<{xx~H3'
    'dRBk@ogkLkYp<6Lnjm25025gjeBORZsR`waQe$WpCyii2mPFl>uwESNL|nPP0t5t(>3Nt>+dy=f_=?34ldOnVluni_J!#&j@S2hc'
    'e~f0e*V+$;nTajcmH<$@~T|%<NfFMW$HhyHxKt8mVKH&?(TjI-'
    '6lgGBe+F2M0f(N8n5?ao;YUs^3KWZ+g>*TF>P`;sTf0HbvY^!M~?HAo?d2j*7fVn&$pxNp%pM57BuPja5$D)J%$0wI1u3fTA!}r-'
    'qxt26K0Ltb=pn#kByKx9CHvsY=q3MNkCUx`(D!qVas<qZ*Y<<G;uc~>U8h969A_>e0ciya<~2tX3byZN-VsX4alwkm?kKM_Am9$e'
    'Xsv5U2W#yZZrPvR`qYWqr1h~G{v)0l48#nQ&30dK!ID#Z!bhjDOWXl%QkZzWsz#${v0LkZ7UQ2F}HH|zsfDG5$%jXP7)rpRVSX@u'
    '`tEVn~c5I>o+7e4I%g*u$y?lmZ)&k@ui)3ktGIX(39U`3@^?oAhG$~+XNiC|1?Uktk#Ph!Z)25T<WrNuF&(7owt7%5Bkz0p7rsdr'
    'vX}TADZ&G7ILBorc09(O{h){QvqIe3?VLOJv+&fCZsq>32>ZI#~M&_!5x(17Gb)Z#Q^f*e{XLe{-Qd<8v!YOcK-'
    'R)byCeRJa`Wj&(1e@m3L@|KaNGAv{}<ke~$xZ%orqhLB5o^%wR4l$`inFW~BN0XY#G%pQfL{rK3R;C>zlTaVAS(W)TYJ;eOLaw=;'
    'oHUuFdeqtHvx$<Lm(;?Wae4LPn^Y`f3{HxmOp;+fJh{4cj9h6#Wx&*v1L(lA?}oHO11&T_^xPb0F2SIng|uZ>^MfN@rXoSFxv;4l'
    '?{#MG<c=_09tp(RQUGmIMC*f}-MG#OI(i3j?n24>S*;lPqMuNcs+hX{x3Vqy&kV{*bC0@ZRfS3~CS<tHtn{bhS-'
    '4}X2tH+0|hkJwKZr*}(n*;mA(L3tkou?bb|!^Vva)RBxvkdw}gpj@EXm7S5JVc8Wt-L&J?-'
    'Xt<7cG?U?($A1hE(xqWj2To!A<iX27e5Ls4p8A5HL<s{{)-'
    ')JGF;1`4)%gdgs#VR&E~MSCM&mZ9XPqRVi!Buh%{U^%iKM=3}f8;`@63d^Ekx4n*6=ue%;-'
    '^f0Hlvj{~=PJwNt$kh>tdi8r&v&&N*>xBDM<4-bDmJHL+K$g~XJ&;Bfzwwbe##`D*gI7o^CcwsEQ-gq=-'
    '?x9$EyfXOTVIV^WWcHn<t?gBv`@mNC+F^DdE<Vr9p)AB?kGHM?D0e~c>b|sD;J^S&fTYP_iG_LO$k0avVVikAgHEFqBN<OV<M?C?'
    'Hc|^W+ibdF9fuF|1Zt3lX>iKrr3DpA=m-?R^(|rfCYID%4IRYUg-'
    'i^NVIrfOg{=w3EUfD#VRsahG43j(Povi>1mlUp0tcSFRHx<}M-'
    '(3=9j%9UerVBo%+P79gpg>mcSXKe&wXdd)<i|CnWIToLY*P0wbJj#sFiZ)bs`SiRf(y_h0vza$~F;BO^<W<f)|Cz8u0nmMf}DXrL'
    '}CCKOT7UlMn@5KZ==C=T5^MfIr$cvYSUeKtOjIhE2a!;m2i`*2t&L-'
    'w?aGBLl|PWsqcHL2zT&)s7}`U&L`ykpUJ<BE>Mzx_~^l(1LUCGo&X)hL$r&7ulCHk8m0C;SRXdK856~iF>o-'
    '?lxvQb+IWxv&{pK7*7uI%8@|}tB$mj9da+hs!u#6pN*z5D}`YiAC9o_oJR+pn69V*&XkstEMN<=Y%FCF7s#y|Ay7C)!J^v#wWXoB'
    'CLdztRz@HvDC~0ud7YBzXJuG0g!Jz!)mwJW;4q506jC+k!IC#-V+o<>xs({cuDdhg&7ZCf!>1qL-TfIbdEhzSZ$Kohrfk=ok)rUF'
    'q$0@x9gF4Dh0ibaO(|&QqTbV>^_7#4T;+q=wHnN>nu6J<4>baq6;x=6Vds}iy&Gp*#W=pog{;No;jBg}0Q&olgk&0QKID+#(_&^P'
    'XEI)?wp75K(qkn$-'
    '~x5oE*n`$KT+dHB?Z#94aH;ofTA%uRIcEY^wvCzB#T<{vA~Y7OBkjy#XNJ~<Ixh9nFZF95iAEA?m|@6;^S$m_uA|)qGxcIdl(zxb'
    'AnH543b5JuZ7(dZnDr>ON|ivw}_+es-'
    'Z8jdJ#M(EmP;UY|>!qH3;JOFE@r6v2VMa;72brwMfI9Y7c*(R%w>;&&G&KG|jReN|d)jf4tNKJ`8ZLGm)R=UgGp6NR2GS4rPw!Fb'
    'bN^Y#5HB1j4eDeW|mOh=q-k?O!+zxx7-'
    '=*&i%oS4!qwR=I+l&uiDpN=Nuvk0lDEAW#nWx|l)iNmWG#1fC>XTVIT$f#V2#C^|5DTBQhkn|}IbeBo_a3%@Vz&jbaM#b6X<V~9^'
    '*3aMNWgH9N503c3XSMH(sK(WjtJfyk~QWX+GpDeEgWW^U7CqKZ)kpLfoVNvrjUg>4qnvMw?Vvy9Gec;1QJ~jal10h7?VTH%d(quz'
    'B)bl7T6eWlv{FH06#CtY=sKLIH7j&+YBUx`+0r!+eQ{I&J=23z#_vc|MEawL%aQIWAawe>Z?FP|ExLK0E+p|{k*?OA`Csw^FB}KI'
    'o$*99I0Gkl0m)9P`|Dt2CN>`qyc-LXmu5Cd2o!SUN>OzDV@*3+rW<hAgD?;=UTjm9@iL;1$p|E*c%;VQQvPe~$5So8wMuO;r<t77'
    'b(uHBcTY4h;_KMiPAfF!BvjzMoFfYFtu8;UUr8Cm{J0zsZS_lW3V8=F+o8BZ#GAtykeAa2yG$0`v`_X2mq7SrK?pnM$-'
    'RiSbCt9_TfNotPdP3FeoipHvk8_bHdfr%D&Wevel&c4bQfUJ?0lZZJea6gnmC8$PxKJf3Ae^ck41GgWvUhmE`|wJq5ML9PJTI2zv'
    'S<Ux={f<ZuN`z*9Y!6J==9)O*NHS4t=L!Y^j|$z3H;H6;q`>i5F{w6TC0y}FvWZZj=QZs3=;i9Cvp+{##mZrL)dzWaF+^NS(G_5&'
    'MAUy6$1No&H$~^%)k>g(OT6xLX0rx0XLmi9#>)&UkW)E(rDTdd&?g)nlWZ@qii0%$X<Fb;<l%9VOwtElmJ31UZ`$(expkjBMXp>0'
    'QjHVtDOg)>d2b<kkFP`M2F!}7IV%(^P~Eci!N#n=^~cPB6w>VMkO`8)wWW(FCkjJsZ1918i#>u&E+DXDk5QDk|15`lPCIsv21ZHk'
    '*GjjT31#kf+Jl;ht4Y#dXG%!p@nplSvk9p0!~W{)^u4GLCASyvE>_})GHlQ()81{@+s0-'
    'MhCr8^4CoHq^6m+)=dX}a;Ab}!w1k1s*}~2CW22S&8bZACoHjyxkrb|WMrCuS?_Mlv@+iw;eW+a`r&=xGYkNpr9Wn?rZ`d=RKngZ'
    'BH4Va#yP?UA!<ncQXYwrsE{XnsF)=2`8R#H1Qk}`FRP?>0I}AL6vPy1NXgK3On9oa=1E6{Yz?W+(!3rU;LVPYZ7%3bM|YcibhnC|'
    '(e#ij$Od&;dP$Si8YM>eB2lpL&1M~2@2{)?2urA#?_PgDNcJ~x0c!ost5LB&EcUj0!-J-'
    '4w%1DXiT_n?VoKB5<WDX`p;?BC4kXEa$n<w1x23w@J6Q=q($QzkqQ>d70(7*s089)Fz_gxAa$9~WN2qdGH~=UegFkS@W`&-'
    'a=ki!61F$_lU|R0|<w0m+fYwyO?X%FMGmoSxgBwm4Mx#>^NJD5{jBx=o0g8eq^UEQ*9yXJe&x%zj3E@E)43Z@Gbc5xCpT-_fTP@3'
    'V<E6B0c~DJ~T*w-'
    'ARBKzub^L8&Y|}t%?(#3zfzmCz?^9S$YjkJPM4!Y{oH@xDbzQ^ly$bn0X(Uz+KyBKwz@64n@ZDRcAET06Km9OIdM;N>O`?%hE0#_'
    'jcbF-<(xPcW1an>EUD4E71pp`3bB#Aef-uWuGlC$!A*p!vm{Go_<+bsOB!(fCr@Fc#%W*bQ{Hkh$Xg8J0TJ1giwIbCy^rc8FEVl-'
    '4B*sep>in(iZJl!dIR70Tp(qZF0JX?ufbdxWPC~f>vEZU-'
    '7+^}3tx^AoIgaI$9b(!e&%uM^ZO)A#pjuTaPUH|;Qz$1IW^1NJ>SP;`*%nX8sJRISVfR^adROL!WRS%rlRK&hNuje-'
    '<ba4{LziVIgP`qsq7JpjCrBr;SpgqQitoDToU*2*<x3s2Q=Xnu@RU<6q|ls!LNd+P@Ekfzr}&AuzG<FxE{wJ`9oRyWK$k9k#EvBJ'
    'h_GT8TBejpVqrL+3@=emNUe<<m(3>Hg)2xiznaQCu(4&7g*|3A2|x&JV@NZUnu8@3qO#jXPxQ&kDAH_}P-'
    'K@jK`xjF4{wjDaK*%ErGelZDTrI3F|f?HZ4f+bAX_%Kwkj6ft=2RbdHV9YwV=i={bt)LB))y@lC@lig>OqMiPKp4sA=F~=+ljG=}'
    'd3>oy~GdQHx4gZz(XKtuSrQ?Gce1h_*>2KsDP(mKs1ecBl#=v%?xXry&{7@XdJ?IQ)6}tSwbcra(np%GFWzIrM=L-'
    'XxHemrp(7s29W@$Gc(rGjT&?#!G6!Y-'
    'J7b>xs&j>AcFvoC}!vC#tl!ZxH(5B&z!`<&QBYk5?Z4y|vPh^r@~PXjSx!fQ7jh2nhzU13ZB)7Ty!6rwKD<e?~Y77F1?SLj|I-'
    '3;<E^DQ1(}*t#5WK~N6zjHYX7ewAyQr7TM<TL(rNb{O6y&qX>g)@(SL`n5)4T#C>(zLHtSHgd~|2p>TrRuYN2r`GpT=^uzxmo5q1'
    'c1oA8Ut+qPe}XXdGAVH<u8i}{y|bg{?VD{VJVCh7oK_2@!kJq|6R3UC-Y#v<?KefKu_s7#2i-'
    'rk_^6WC*4|6Br%x0YcS>WE9z2fRaw3HZJaVO84UZt~+_P(F!AvVEcQk22%ZO60AY-'
    '~~y~+yKO1{?#rouhcDf=1XKeW2CvZfW?bu?a_A*<{=sqZov3wq6~)((B{tJv`fB1wK|h2zh)DAF^uT8%zlXCkzk4`Os8F=w`s<|6'
    'FHn0><Y$xOXRlrr*`>QMPDX2xOj4M~~tX{#M8>2lQx`h^ojpb|V|!IRq3AvwZCns^=y(+li+3l`V2_O;yaQM;yC7ATz!o?hQmTzv'
    ')h?g^Oc<r35d`RQ_d>s;bD&SjR9*KIYgaSC`XrHNa6M3-'
    'tnlaV|{zoM!4F4f(+{c+<q*J52n=M*&SM@*q0%Bftay}3j<2vjM_E0l_jEL3%Db__~VDpDQ3b9|y4H?QjeX>B<&^U@=I8watPNAO'
    '^>!~yeI-F#)IX@DSwoUyzl!;AhH19<U80sDu^n4xWHvmy?SL7-'
    'VAN_lDD5xFa`Smq(nJp<WKZBljZD|9#<n;+DAdeW}#nl$N><5)7uZ9dT#7m=ELg5NQ#78tsE>uqY&kXuw3EX(^+w4c=|p*DA!AaT'
    ';A=D47EEoIN)MoTmu6aA#rl-zj6E<hA-Q<+Fv(JH@lTC?@wp(>T}2en<fJ(m!+U=5sza`Nz@wsEF_#&$8)<oLjwerP^*(%gw-'
    'D`W}%O5MRf`4e%8{9+Ie%x8<r1KHxV2DE@=x9n3+;&1xOY#DK;`3u~uSxyFdXD-'
    'w+ov}8Kxz(UB1k)u>2h(JeSiw5b3jIuPmL|wXY6OMz_7VQ%B2k^1eX}5%>ffbUNoC%8lW3ikRmD+pep&h!jR5UCO^JZc8GNe*IW_'
    '6$NoY3=nc<>nyoD1q5LXL1bZU8A$FNvWa9%OaBH!Sbr4_m|aqw!J==5zNc0Ce~j|aMPei!kuF8JE^OnFsrVoqO3k+uT<D<jA$&ca'
    'r}Ktqa~TY<p&Cr!<cybL>{xA$%K1K_!6?XU%jaz=pSe0}`3ebUt&?*%JL`vFwiWV@vZtM)at>=ReaD8c$GsfUDf(M25#3NN}mq|j'
    '@+0Up;<Z@XA?Bg+h8IMCBKDP2^C0LhTOZKFePS6!J&t1H}-y;-'
    'Kht1LQ&w&wc8V3*Hf;l(&tD#M48qs$d0>~B>tNBK=d#2rKuS5N}9X{Lcy$x%e9QfPqPr>?9r6T6EoxyWI*0w5@^G_SYOt0Q!bCB@'
    '6BP%r`N#zHZ#K42Xy3$vRnmObYJtNSk9inoak=Wz=nAQF1d$CuFA(2`ncLvSB?MV#-'
    'mQv@d|4wcnz?b@P%?O=+iv62P0bQb<tPU|0bod6^_E*$FpBn~1)Srjd^q{>N(qN~xe+OZQIkz9!%5m2EOU@R)piemYLNV*jZ8ajI'
    'YSF}RMzl-17wYI0|tsK<L*d5?bvD~jyFC=6_MigI!NWw07<x#=qM2RK<5@Ibdx41moJbSGS9blPN!?QDmnmcCBLhz%YffK-'
    '}8fQw7j1C7sH&KtuUeDo|I9?$TmHaw|BW(?-UZEb)WQpDZXH7!KwXB#^STu`Ac@H!J-'
    'DF@aqme{bgYe&MX=zbeR!05w)l%~n+s8h=FsGOHiX0Cy^hFN_+vQ0ZZ0p~h!B(!)TZb8jWEEV;t~#`wQe%Y)f9lp!$HE<4N`R}wV'
    '>O`#TI>J~R8ba&Zkx_McEAn@c^}$9TY7~cN5*ag)tpK%_1dxrf{GBe;wFs{NhJzq#4>1Exq){ifP?>^*c0ipC7HEEn%9SrOllL0('
    'I}ftOdJTlLHPsv_C}AjBOxqV-UA#o4RWcgMXc2uv9{$WFe*(h%hb@a)+nBg9zh}&l#)~?T|yhgmN*$j9D7*gq#wtW*3XDASu>kv0'
    'e~aTpJVtugjIV|HGnqJ1i)sO#71+Vvy-A~Bq1xLd$|p$e%#&Nzeh~SbVU?A;8``ht?&r-'
    'oT4~a5LXKQNn2h9U%&Qi^R~VCPwxN1v2Rv(-PY-45^A&~r~tq&YWm;xu2U{QDha}w{QxwP*nXk}a<epbj^&o{aEF;Fjtjw};-'
    'zY$Zf!Ah%Sq8L$LZRfgwwTonefjl!E*7LORPInFPhe6Cv-^Am-hejVrg8o9*&qL&*MsLqdvSYsVYwKRnsi4R2jl!7vu`e(Q$TJ{S'
    '#a_KW*Q2+xb*2k0GLU&h=}}YDHO_$EGgS$s$t9UZP=cjlP!QG|@iQOv2ZLKvdb}242(UV_V~EDDwtmNviimIzjc>rLTkv3#}Qotn'
    '!`aK*j5Lo!F<+I;tQ`yt+PI!WR(E=K3+w=!tV~qQX@A$z{}q%))BV?uKr`@hK8Ncr_ZGcHT*p5GgMsfUu=T2|*Eh;tXe{O#$g6ju'
    'ufra;XhD+)1hg9D5e*L)Jw#&D+~yy%$q=qz;FlywD#Mg}K5{U_YG_&6nXZG`%@PF|X6IW-{Uke}oel-'
    'm0^?3B?MlUaD1klP{eaMc$}TMK1N|NfbBjeA1dpy5N}|e<eYXTbFP-xHqIXmP<~PB^5*!`cSUy`O=#UtTaD+_#NWdMF;fo@+2r>X'
    'PGk*@NAbq)5u|Ebg&8e1Da<P5b^=8Y~^1QQ_L3s1)Zeo6^c#Ufrz4qJcPD!<~j_9l4cAg$SS#)*kOSKV(}VmUs4*GVH#*mF~ksz{'
    'ind8XC+i}=*oy=bPKYFQs(4&Qz&=$mD^DTZV(W4)x?G3D&KhB!DIC1-'
    'M@OCBo~(fbJJrTQp~ICN!v9}C?rPd<_xJ*?F+57QLNj)@(MhxKYRWXd`JyYsPJT2t7~su0W~JrB4Y5>rHwf=|6z&3nc)M;=UElyq'
    'a9#~&4rf*=E5;f(8z+R_>2$c8n1*<b5u~YXZVz;Ma76=r`1je%&ZDC)sp?XDD(@eGyw_)76b^O;nS$SCEe<#g{j7r+9D!+-'
    '+p_sS9v%d!x#BTfHXc2-(oETp1;UqT9$3<cf-'
    'V!+M$)zT2<R(e6!`0x4Lx=B;Cd!=&*JFbJ<GUTJuhYCso(7#ZYP5*7`+gQjsv9b(HEt%W9j-'
    'rlPmJ&y^a`?m)_3?!&@@I1p9I(mE4Tku<K_+pMcadi^n$53NiT6u_5P)=ITflvz7u2M1MWX{B#jHB0N2zR$H*r_G!hA5fj|s*PsG'
    'fYvCsleGASjk<2)CMB9wBbuY6z>!o)s}bt7ng#Y?rn&**)n%x-'
    '((_#<(dDGIoMICc0zNzTVyg#fzs7rzg%gp*bL^N3{HbP)ov_zpn{3Dqn~CD!Kr4pMj9DOYsID$DWI<<Q43y+&ke0L{zpS$cJ;nvE'
    'u=NU<0wl6lW~-'
    'w}(T*!HXy~8XG`)DHAS$aRWtutFQm*n<VbmPXKtT;Id^bsg`^qAfO5l`diO|lI(qsDwQck3_=6~YhK@2vBbq3KkJN}(rR+4*^gxC'
    'xtslCr`F6RL1%LACy8bR!;5bou5UtRw-'
    'CwvX>P}9+;10Z?z`&EQ)7`rt;Hw9ExIvW})RC6(DLvzeBIZVPA)5U_zvBCf_XmOo9D@-ZbuMvY-JljAI6S9XjnwYd9g25#kBV~N('
    'RxFv5CrqHq+{!+CgXV_=7B`8^<T|N@Y_%l+PKrH)q$LFfT!bYxU<P{f9Zp?cyiCJp(sg>NGB_Bs-'
    'QOW6h_z%QTU_5Js1kE0lE`9#!h=zn5S3UONIZ%wp}aIW$`aKHI$_S%AGpw9ZkOf!YY(50o2;1t11)i4MyP~tC-Tb7j@qk~(Op#S('
    '?dm^9z1ars?I?lxhzY)*JEYoAR(F^Itm%7g}0{GtE$3Eq`U?mRY=_L;+~7mZ4=S>C<%;l)nax|*d~0&v8h3Il+Pxso`T;NSZr72)'
    '|TOZlVgRXf?Oxop>4m|PoG`nLtA2Yfh}g|Aq&p$eWg7160Z##V@6wp25q{6Qrk6#n;dq~g3P$fmxEB9UbCz>`R$t&DqDqot8Yq&E'
    'Ikt+L|967OrpEh=dldija5W4sv8&lNzieY+Exq_mEN6HnoDoJw<RxtlE4^meMlF`(obT@iOp@J^W0dqWgI_-CE0R8Xei#NHig^@<'
    'S)K1QomJ^<dsHu()w_b(vzU?N7jV#xtu-'
    '>T}8zRkqyZZhEn!r^{v$RbI?}Ujy9B(q&3DEY8ipLR96p1bxJ^sa=wKSrX&=srZ2_{5!vg9@vyYb#lBVQT!<S=3se)E97uN%VXIu'
    'kO6__`>I0fc0Gf=)*l7W{jAdg<x+pE_+sA#VQcpR{ywE<I{28W2CbaIPhwxk2d55ybG;LW%Q7WuMpshy-JRJGq+l9_Hq*HBTidb0'
    'cDbNAo+e}ww0Z?+LU8LIR#5TOFwE>VjBmh&$jUo@>>9vDit-'
    '7LeHwKh@g*yv?x{pSD?)v4l4(Lx5Tc1QGtSl$rkrxKJR2du&Dx7u3<O&$cP?_GL(p0J0Y}6DZj#paVU&}W3))-'
    '^YgQ%X>HeALU3zFAtlTc(W6=XGasDz;!hslCEvxK;INh%wjES89o3#ou`O32wuNv9<=RGz0|2QVxela%HH+->VYFdr5j2g-'
    'p`Wpcz|sfm2kU^x;+3CaRceejTd%}dz<G&t%mTLXBQ#rFiGYFq(IrRCJfu#M|3r4gNDTltSGIkKs3HN&@w*=ewOfIrc_&c?2X*<5'
    'QP>R`H0ky@x|GwWPYCAr01JiP%dVV@`C)3E*M`XY`40=FU|y1lRR_F#k`vUT=JWqaF!(^7OY{UX~{!nQUp-'
    '_}(ZvNep50cTGDBJ7qYV;UX`$k-'
    'JsLRq8~6%X3^rKI5MLU(q`u60y#fMG82(F@*A<B}g<l#5MX6v$5@VVjjwH0h#pTdb(xQSR&y!AZ;GI?c6?oe5>hF)oTxDJ4EMy>k'
    'Mwd!<QOeT;=j*4k>+KbQ}mm9%H%vQC9MvBEm4dX*96m+BO)G_^xjq1t3(ooE9blOtLYDgq__0p~ckXL)?)Z(yyhrlJN{<kMKw@yv'
    'l<phq>%vRI>$wXpj)4AEp@LlXTcwXWnWQ;{7R)xwoF1TqdP+6vI+=n$cdi0r1=OS;ezqDZ#_Z|qS+dXQNo3{7nbrWNUSFf^J(0E4'
    'Q9rJ0=!FhRLENyPrR&OV!SV?RO}R1-C|0GopmQJ<sIb4Z+4h!1cpDMCe5B|}r^*ZUag;6-zVDS<pyO?Zi6oKxnLJNU7-'
    '072m*m>$QcXh%MSHylefSvJg7)0T|@F@vW>XsAxR0)!|2o3f`}u=EySSlG@t;BmTaf?w>WO|;-'
    '*TC#@zB7zhaaFjrYwze64TcpNWkV<Go;P{56SvM>C?kllLrp-sVA#;{RMM%X?t5Tc$@NK8cB_{&ajCt|Rh=bZAQOGrTGTd05h7t('
    'e>aBX2rccNorthpCSP92%%Wpl_QQoEuiQCL8?IgZbkg5_{$(&x!U&M-'
    'E3qmT8M53(<quegI1<6C&WeSr~H?Fh*pi0Mu9e!HO8+I;Bcp1Al%O&efQ=Y&+L`I`Fwr~CVHV66mkH`N9*sTax'
    )
)).decode("utf-8"))
_PRICE_FLOOR = 1
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
_WEED_STATE = {0: {}, 1: {}}
_WEED_REPLAY_STEPS = 8


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


def _impact_slots(obs, action):
    action = _copy_action(action)
    market = list(action.get("market") or [])
    rows = [
        (_impact_score(obs, order), -index, list(order))
        for index, order in enumerate(market)
        if _is_sell(order)
    ]
    if len(rows) < 2:
        return action
    rows.sort(reverse=True)
    ranked = iter(row[2] for row in rows)
    action["market"] = [next(ranked) if _is_sell(order) else order for order in market]
    return action


def _v027_v22_original_agent(obs):
    try:
        step = min(max(0, int(_get(obs, "step", 0) or 0)), len(_ACTIONS) - 1)
        action = _weed_repair_action(obs, _copy_action(_ACTIONS[step]), _ACTIONS, step)
        return _align_hands(_impact_slots(obs, action), obs)
    except Exception:
        farm = _farm(obs, _seat(obs))
        return {
            "farmer": ["PASS"],
            "hands": [["PASS"] for _ in (_get(farm, "hands", []) or [])],
            "market": [],
        }


def _kaggle_submission_entrypoint(obs):
    return agent(obs)



# V027: product-level quantity-preserving sell-wave shift.
_V027_VARIANT = "product"
_V027_STOP_STEP = 672
_V027_DYNAMIC_CUTOFF = 648
_V027_CHECK_HOUR = 6
_V027_PREMIUM = ("MELON", "STRAWBERRY", "MILK", "WOOL")
_V027_SHIFT_CONFIG = {
    "MELON": {"ratio": 0.25, "max_units": 6, "cooldown": 8},
    "STRAWBERRY": {"ratio": 0.125, "max_units": 6, "cooldown": 8},
}
_V027_STATE = {0: {}, 1: {}}
_V027_STATS = {
    "calls": 0,
    "errors": 0,
    "action_diff_calls": 0,
    "action_diff_farmer": 0,
    "action_diff_hands": 0,
    "action_diff_market": 0,
    "planned_sell_units": {item: 0 for item in _V027_PREMIUM},
    "actual_sell_units": {item: 0 for item in _V027_PREMIUM},
    "planned_sell_orders": {item: 0 for item in _V027_PREMIUM},
    "actual_sell_orders": {item: 0 for item in _V027_PREMIUM},
    "shifted_sell_units": {item: 0 for item in _V027_PREMIUM},
    "future_reduced_units": {item: 0 for item in _V027_PREMIUM},
    "shift_events": [],
    "future_reduction_events": [],
    "price_events": [],
    "mirror_checks": [],
    "mirror_latches": 0,
    "mirror_releases": 0,
    "mirror_active_calls": 0,
    "pending_missed": 0,
    "pending_at_stop": 0,
}


def _v027_new_state():
    return {
        "last_step": -1,
        "pending": {},
        "last_shift": {item: -100000 for item in _V027_PREMIUM},
        "day8_match": None,
        "day12_match": None,
        "mirror_mode": False,
        "divergence_days": 0,
        "last_boundary_day": -1,
        "last_prices": {},
    }


def _v027_state(obs):
    seat = _seat(obs)
    state = _V027_STATE.setdefault(seat, _v027_new_state())
    step = int(_get(obs, "step", 0) or 0)
    if step == 0 or step < int(state.get("last_step", -1)):
        state = _v027_new_state()
        _V027_STATE[seat] = state
    state["last_step"] = step
    return state


def _v027_opponent(obs):
    farms = list(_get(obs, "farms", []) or [])
    seat = _seat(obs)
    return farms[1 - seat] if len(farms) > 1 else {}


def _v027_tile_counts(farm):
    counts = {"COW": 0, "SHEEP": 0}
    animal_yield = []
    for row in _get(farm, "tiles", []) or []:
        for tile in row if isinstance(row, list) else []:
            if not isinstance(tile, dict):
                continue
            animal = str(tile.get("animal", "")).upper()
            if animal in counts:
                counts[animal] += 1
                try:
                    animal_yield.append(max(0.0, float(tile.get("yield_units", 0) or 0)))
                except (TypeError, ValueError):
                    animal_yield.append(0.0)
    low_output = bool(animal_yield) and (
        sum(animal_yield) < 0.5 * len(animal_yield)
        or sum(value <= 0.0 for value in animal_yield) >= max(2, len(animal_yield) // 2)
    )
    return counts, low_output


def _v027_mirror_signature(obs):
    opponent = _v027_opponent(obs)
    hands = len(_get(opponent, "hands", []) or [])
    unlocked = {str(item).upper() for item in (_get(opponent, "unlocked_quadrants", []) or [])}
    animals, low_output = _v027_tile_counts(opponent)
    match = (
        12 <= hands <= 14
        and {"NE", "SW"}.issubset(unlocked)
        and 7 <= animals["COW"] <= 9
        and 5 <= animals["SHEEP"] <= 7
        and not low_output
    )
    return {
        "hands": hands,
        "unlocked": sorted(unlocked),
        "cows": animals["COW"],
        "sheep": animals["SHEEP"],
        "low_output": bool(low_output),
        "match": bool(match),
    }


def _v027_update_mirror(obs, state, step):
    day = int(_get(obs, "day", step // 24) or 0)
    hour = int(_get(obs, "hour", step % 24) or 0)
    if hour != _V027_CHECK_HOUR or day == state.get("last_boundary_day"):
        return
    state["last_boundary_day"] = day
    signature = _v027_mirror_signature(obs)
    if day == 8:
        state["day8_match"] = signature["match"]
        _V027_STATS["mirror_checks"].append({"day": day, **signature})
    elif day == 12:
        state["day12_match"] = signature["match"]
        _V027_STATS["mirror_checks"].append({"day": day, **signature})
        if (
            _V027_VARIANT in {"mirror", "product"}
            and state.get("day8_match") is True
            and state.get("day12_match") is True
            and not state.get("mirror_mode")
        ):
            state["mirror_mode"] = True
            state["divergence_days"] = 0
            _V027_STATS["mirror_latches"] += 1
    elif day > 12 and state.get("mirror_mode"):
        if signature["match"]:
            state["divergence_days"] = 0
        else:
            state["divergence_days"] = int(state.get("divergence_days", 0)) + 1
            if state["divergence_days"] >= 2:
                state["mirror_mode"] = False
                state["divergence_days"] = 0
                _V027_STATS["mirror_releases"] += 1


def _v027_overlay_active(state, step):
    if step >= _V027_STOP_STEP:
        return False
    if _V027_VARIANT == "melon":
        return True
    return bool(state.get("mirror_mode"))


def _v027_order_quantity(market, item):
    return sum(
        max(0, int(order[2]))
        for order in market
        if _is_sell(order) and str(order[1]).upper() == item
    )


def _v027_order_count(market, item):
    return sum(
        1 for order in market
        if _is_sell(order) and str(order[1]).upper() == item
    )


def _v027_find_order(market, item):
    for index, order in enumerate(market):
        if _is_sell(order) and str(order[1]).upper() == item:
            return index
    return None


def _v027_route_quantity(step, item):
    if not _ACTIONS:
        return 0
    route_step = min(max(0, int(step)), len(_ACTIONS) - 1)
    return _v027_order_quantity((_ACTIONS[route_step] or {}).get("market", []) or [], item)


def _v027_visible_inventory(obs, item):
    private = _get(obs, "private", {}) or {}
    total = 0
    try:
        total += max(0, int((_get(private, "shed", {}) or {}).get(item, 0) or 0))
    except (TypeError, ValueError):
        pass
    for inventory in _get(private, "inventories", []) or []:
        try:
            total += max(0, int((inventory or {}).get(item, 0) or 0))
        except (TypeError, ValueError):
            continue
    return total


def _v027_current_market(market, item):
    inventory = _get(market, "inventory", {}) or {}
    prices = _get(market, "prices", {}) or {}
    current_inventory = int(_get(inventory, item, 10000) or 0)
    current_price = float(_get(prices, item, _market_price(item, current_inventory)) or 0)
    base, equilibrium = _MARKET_PARAMS[item][0], _MARKET_PARAMS[item][1]
    return current_inventory, current_price, base, equilibrium


def _v027_next_future(step, item, state):
    pending = state.setdefault("pending", {})
    upper = min(len(_ACTIONS), _V027_STOP_STEP)
    for future_step in range(int(step) + 1, upper):
        planned = _v027_route_quantity(future_step, item)
        already = int((pending.get(future_step, {}) or {}).get(item, 0) or 0)
        remaining = max(0, planned - already)
        if remaining > 0:
            return future_step, remaining
    return None, 0


def _v027_apply_due(market, state, step):
    due = state.setdefault("pending", {}).pop(int(step), {}) or {}
    for item, quantity in due.items():
        remaining = max(0, int(quantity or 0))
        index = _v027_find_order(market, item)
        if index is None:
            _V027_STATS["pending_missed"] += remaining
            continue
        order = list(market[index])
        current = max(0, int(order[2]))
        reduction = min(current, remaining)
        if reduction <= 0:
            _V027_STATS["pending_missed"] += remaining
            continue
        new_quantity = current - reduction
        if new_quantity:
            market[index] = [order[0], order[1], new_quantity]
        else:
            market.pop(index)
        _V027_STATS["future_reduced_units"][item] += reduction
        _V027_STATS["future_reduction_events"].append({
            "step": int(step), "product": item, "quantity": reduction,
        })
        if reduction < remaining:
            _V027_STATS["pending_missed"] += remaining - reduction


def _v027_maybe_shift(obs, market, state, step):
    if step >= _V027_DYNAMIC_CUTOFF or not _v027_overlay_active(state, step):
        return None
    if _V027_VARIANT in {"mirror", "product"}:
        _V027_STATS["mirror_active_calls"] += 1
    products = ["MELON"]
    if _V027_VARIANT == "product":
        products.append("STRAWBERRY")
    for item in products:
        config = _V027_SHIFT_CONFIG[item]
        last_shift = int(state.setdefault("last_shift", {}).get(item, -100000))
        if step - last_shift < int(config["cooldown"]):
            continue
        current_index = _v027_find_order(market, item)
        if current_index is None:
            continue
        current_quantity = max(0, int(market[current_index][2]))
        future_step, future_quantity = _v027_next_future(step, item, state)
        if future_step is None or future_step >= _V027_STOP_STEP:
            continue
        inventory, price, base, equilibrium = _v027_current_market(_get(obs, "market", {}) or {}, item)
        if inventory >= equilibrium or price * 100 < base * 105:
            continue
        available = _v027_visible_inventory(obs, item)
        quantity = min(
            int(future_quantity * float(config["ratio"])),
            int(config["max_units"]),
            max(0, available - current_quantity),
        )
        if quantity <= 0:
            continue
        market[current_index][2] = current_quantity + quantity
        by_future = state.setdefault("pending", {}).setdefault(future_step, {})
        by_future[item] = int(by_future.get(item, 0) or 0) + quantity
        state.setdefault("last_shift", {})[item] = step
        _V027_STATS["shifted_sell_units"][item] += quantity
        _V027_STATS["shift_events"].append({
            "step": int(step), "future_step": int(future_step), "product": item,
            "quantity": quantity, "price": price, "market_inventory": inventory,
            "planned_current": current_quantity,
            "planned_future": future_quantity,
            "estimated_unit_prices": [
                int(_market_price(item, inventory + offset))
                for offset in range(quantity)
            ],
        })
        return item
    return None


def _v027_sell_totals(market):
    quantities = {item: 0 for item in _V027_PREMIUM}
    orders = {item: 0 for item in _V027_PREMIUM}
    for order in market or []:
        if not _is_sell(order):
            continue
        item = str(order[1]).upper()
        if item not in quantities:
            continue
        try:
            quantities[item] += max(0, int(order[2]))
        except (TypeError, ValueError):
            continue
        orders[item] += 1
    return quantities, orders


def _v027_record_prices(obs, state, step):
    prices = _get(_get(obs, "market", {}) or {}, "prices", {}) or {}
    for item in _V027_PREMIUM:
        try:
            current = float(prices.get(item, 0) or 0)
        except (TypeError, ValueError):
            continue
        previous = state.setdefault("last_prices", {}).get(item)
        if previous is not None and abs(current - previous) >= 10:
            event = {
                "step": int(step), "product": item,
                "previous": previous, "current": current,
                "kind": "cliff" if current < previous else "recovery",
            }
            _V027_STATS["price_events"].append(event)
        state.setdefault("last_prices", {})[item] = current


def _v027_record_actions(obs, planned, actual, state, step):
    planned_quantities, planned_orders = _v027_sell_totals(planned.get("market", []))
    actual_quantities, actual_orders = _v027_sell_totals(actual.get("market", []))
    for item in _V027_PREMIUM:
        _V027_STATS["planned_sell_units"][item] += planned_quantities[item]
        _V027_STATS["actual_sell_units"][item] += actual_quantities[item]
        _V027_STATS["planned_sell_orders"][item] += planned_orders[item]
        _V027_STATS["actual_sell_orders"][item] += actual_orders[item]
    if actual.get("farmer") != planned.get("farmer"):
        _V027_STATS["action_diff_farmer"] += 1
    if actual.get("hands") != planned.get("hands"):
        _V027_STATS["action_diff_hands"] += 1
    if actual.get("market") != planned.get("market"):
        _V027_STATS["action_diff_market"] += 1
    if actual != planned:
        _V027_STATS["action_diff_calls"] += 1
    _v027_record_prices(obs, state, step)


def _v027_base_action(obs, step):
    route_step = min(max(0, int(step)), len(_ACTIONS) - 1)
    action = _weed_repair_action(obs, _copy_action(_ACTIONS[route_step]), _ACTIONS, route_step)
    return _align_hands(_impact_slots(obs, action), obs)


def _v027_fallback(obs):
    farm = _farm(obs, _seat(obs))
    return {
        "farmer": ["PASS"],
        "hands": [["PASS"] for _ in (_get(farm, "hands", []) or [])],
        "market": [],
    }


def agent(obs):
    try:
        step = max(0, int(_get(obs, "step", 0) or 0))
        state = _v027_state(obs)
        _V027_STATS["calls"] += 1
        base = _v027_base_action(obs, step)
        _v027_update_mirror(obs, state, step)
        market = [list(order) for order in (base.get("market") or [])]
        _v027_apply_due(market, state, step)
        if step < _V027_STOP_STEP:
            _v027_maybe_shift(obs, market, state, step)
        elif state.get("pending"):
            _V027_STATS["pending_at_stop"] += sum(
                sum(max(0, int(value or 0)) for value in (items or {}).values())
                for items in state.get("pending", {}).values()
            )
        action = _align_hands({
            "farmer": list(base.get("farmer") or ["PASS"]),
            "hands": [list(item or ["PASS"]) for item in (base.get("hands") or [])],
            "market": market[:10],
        }, obs)
        _v027_record_actions(obs, base, action, state, step)
        return action
    except Exception:
        _V027_STATS["errors"] += 1
        return _v027_fallback(obs)

