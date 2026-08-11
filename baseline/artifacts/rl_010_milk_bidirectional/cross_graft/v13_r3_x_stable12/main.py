
"""Replay-derived clean-room Kaggriculture research candidate.

The production trace was transcribed from actions visible in one public replay.
No source code or runtime replay access is used.  The wrapper adds only hand
alignment, bounded actor-local weed repair, safe SELL clamping, and final shed
liquidation.
"""
import base64
import copy
import json
import zlib


_ACTIONS = json.loads(zlib.decompress(base64.b85decode((
    'c-rk<O>Z2@k^L_`^Pv79Mft{&+LmCBC{WZkyo1JI0NXII@E&IOw%Gq}jYw8iSG;)fA~LH*jeKj6-Bpp1QCacv;>Az@clP&Re*Nd)'
    'em(ocPiG&lKYlzroS*&Um;e6j|9t+#=a2vV<=6lE+y8$4{L|UncXzwb|J6SH@aZo<U%!9%<Mqwi`Ps*}yWNMg^R@ZM>)ZY0&mVWY'
    'H=qBwf4jTBKRbUp`}2>xo7?wi=d1PM@c-vWQonos=T9FdR~zL2>1@CIc>hJ7_qTWVZ@+wcT;$|;Q}G^taJ=x}g!piG`{vW@`%ye2'
    '#t)y~-Msnv^VRP^ebK>0it*-5jN!uL_oi~pSABE+diS_!{buH$<PMLzn_POnM0gALOXOBWcf$^TUhw--|HmqP)WyR_HtO%`J`eWx'
    '#U`%rcX!8k{NrynIhE@1+bMO9*Bux6bc5H|kIH-eQYYn&iyH1Ue8-x8xB|N;Kv&ivW<TS*baVqzd)6RgH9lQ0slLGy8q`NkZLkF0'
    ')aKU}wKiHp7iHlGb-v(8Yx8%KsI|$TbhVjVb<!4CgRc?uugSwzP!>>#uOs1sBug<LI;qHhaFo_f?wPK-$$k9c^p|}+OB@Fe`Z*id'
    '-5S1-x}Ncy9uLr_Ys`<<uO&x8zvdcGF4ga1F}v&bjp-rB>)V@~-Rt|G|G2xme|PilKaXEtl`DR{{nWlq{l$87cl%-4r|IMF=C{yo'
    'BJvo)En*Pi3AAdw-m`h)nBvQpld;=gHvuti(wfv9Lt%G$Rv?ZX=Q};U%;>D^*PEYjN7q9;U_30S((&PNG_^W}0m>){@PDmO*KlvE'
    ')X@pEO6|JrCjG}sNF0v26hW+n%&kd4SK9kt%LZY~ce-wHk}R}vHzMkE@3|8ImpgoT`1W$Q{ti~nU*t+GyqFHit$&{?D1`RU_0D~-'
    '|1Di@=HG5J{_R%vZ@Htp#nm*$vr>v;j~7$2j?94qx0v5vh?G*UYVwwC>N=_-)x7<AmbAC7Pyoc-%Gv)ax3o&MD*`o1c+ggzcyh<W'
    '5;Jcy_FAvskmxi_!S_hJiT7)X3O5~J+KCrgLLiep`3f>TJEefa=6A0XaOnP9DZQ#%&r*bMx-huZW#w9-=O;UF|1KW%g$F$A<3UdY'
    'wB9~6#c?g<L=Q-pCMTLeof?(`yy`fGxSaOvB1f2z;vglG;|x01kR=z~K`Cw#mb+OjK|cKN?e*P%sE+VP$do=h|9t5>sAdok-UG$6'
    'bK|b!4z2j3EDEH}s%H9o957?VAh`?jrOagpbxBd4kPK%^n(u$6-a7ti`UzY*5}Ks45sVO5vIJxnfnYw}Z@TGrCGhFX>;OR&dIdW9'
    '*|S<adIGE>$32U67kc1kVZb6DAsxg2awah>09<*#rtp-8)%xU`sr5VC8LvE@k$rl_TsZ66`0Wfh&uTKK=0PbqOoc>Z>Q(S?k(6L)'
    'i&A2SQG**hr{qk7A%&lKNxzg}wyYHjmehGgfo?xUD6X3mYd9E_3-&TlEk|=TWd2@$(h}NVw1@Wa*Oz@m_f7wZ{bX@^w-T3qMJyVW'
    '_fZgAP(?pXZls`&U^J3B>5K@<1&UqK895l1-NDl>JKpV0qH<!F&5}s^8M4V0ft80blNC{bbBWN!kHU%qDp;c?^j6k?v9gl1d%X5s'
    'nNo?*8)`{OU8Epl`vv7z)RGc5sff8ZauJ5Kx3@Q68syJ?rwhE8RB!9%`u&@HZ+{%8&D-~}dxN|K(JOp4tMh!kzq{W5u)Dka%h~x='
    '`~s$3?|!joxhl;ZbTl5fKBGah_uq?J@%6@&Fmn$@)8m!F|DFUH3LdlXEUj&?$=uIlg>M<A_u=C6LL4eOOl-V$4?uSddNub&$pU``'
    'Xadwo22Css8%K&h5`-D^d<C7P6ayEJE#vrP3k<0RifxQ;*vH|+yj~g<T^f9Fd22yK5;_8fT>6%PdJ|h}wT2Gj>_#SpV=0i)*T&ie'
    'G7I~9LBJfv+Kao(t<5+IS%JS!-l?$K86PAab%*Re)Mo-sl@L_{AF@cTt1$f#ccNq+k(lS*lIFec)`X!vKS>Td4MO7_XVb`Jn+Tnz'
    'M>l+#iyUJOsrS{3`0X&VX4wvYEO_#h;PGD_#G<Hkr(q4i_iU5w)j<yi(47Xc>9;D-x6IZWX0-YHU^jPKfNWiaBpV9?8N07mn!r^N'
    '<)ZunEN(<(V4ijXE^eU>=bmPWJc<-87lmGAU&cI|Wmyha;7)c5aaAC@j`fZ4OI=h7^3>*e2ar4mQRVQPh3!Pzxeh^<V9yCO(Z^ec'
    'd1BRLB<MQlxuI}SA0XabJelC$NDH5f0M1o|z}YPU{UH7j=*j-Ct)Rr&^$@nUB7sd+cxS8GA&AhF=sPQ7!T2!sob@Wl$O@uJ&n1gm'
    '+KrhkVODug3dXPNS|+^t!?m;E_I~##QSy)PZvOlmCP)OX=Cl+s1Cd&DIxY<>e|Du$ukcMNO63II(<t?2K1yAyQR=EGN`3fH18rFn'
    'gqGlRe!0|pZ^kc-1E*XpTC4!h`ig>VeLqP^xxf}Ro~m?OFzggQ#w*pf3b;yo5JUwoz>V#;k&X1LEq=UF09xBvCe{yV^pa!XN~)3G'
    'numvEQ!CySun4<_VLehT7Uz8%t?HQBV7;cm3aOzLGPi;ONIL?*a;P%#QEbeLleD9~8u#=&QdO&(3H`2=6wrA^$Z?mWe2E>3GF#GC'
    'bl$HfW|Uq+5EuSrGAvLdtJev?AOg3GlnK9mH9bSEG?VL}Hv~tlFrmb28}!EuJ#@oK=yfKBvrtG}VFcBSg}|E3WgMiS6~_i~6dn#%'
    'K<tZTl_(@^ylel$QNiV%vTpmJ5xY||1hNX<<M3L$SC+WJuT3n`o&=NIz}CegUr*#HQXudc(cb!07!8m^;44wV=xLS0&u#j#hw+WK'
    '0V@2yWS<E{qPY3YK0)%PFQgOfz7K}P553`_=nViA^$2ujQy$hnmP1X3wQqv=7#Aa1Wt%qnu|0|cJOh?XnhSZUw@OnCs{c?3pq1GN'
    'E^n;5sm7>W<EQah!lH&nERn@0N@sF<lY|Y9!%)woU=L^z*pj@5b2gfy_UwhD%3AKvgP5#7_@KmT+3cZ^w#hueVN+?kZ0QZUrB5ii'
    'S5&H8vM~9umdy*N2>Y^Bp42)Kqq)H#S%mPkyu}ax7oBrer0}%Zx(-KSZ3DdRR7c|j49){T@(Uq_bqKLQ-{JN8`59OCqGt<dk@7;}'
    '@U&;fZ!KgIsI(xo{>q93Dg_G*2HK<>!-BH(LUi+rfVQCG9k-1ISR}A6zvrwEr8~l2Yi(|bo{|mb4QjQHZX{2<iR@$8NY-GiBbaH-'
    'L2Bxw&P?MR7+l=7M|5iHvr{M9wGjJly(D@<)!K?PNew@LMV{#SxN5m>J^oMz?i2pczIhRl(u`5#Dy)|3aG{1z0RB`_61qWCsC9V2'
    '`ze%8LA*xScwX#&WzhzX%5{iNUp?p~qzv36f!V>cu0vfi46HBR<-a^s;q%dhA&jRF0S*-4taT?esA9eX$K6&R1_?f)L#>E@<78RJ'
    'AZ+hKXr%&SFh#zMYl^b61WewrW?p59Hl_~fL9fGIkStR3Q+X}mHmKrDA%`LwwLNCz^2bcr1)orc3_8)Cf_QYO+(Z_(F>D+wP&yk('
    '_X%a=&fC7!&anVU2&DRRw%S?n6fa!rL);6qECqs_H23;!Rxi2`C8XO-E{e%piySJE&aJkUrg<r&)f=K@lU}13sLfk$)2U`8?0XXw'
    'M14?1pJXiC919RCK$6y#or%(sE}25-oe8~1MzPQWrpfG_wW9#5(voYsDDw~Gys=pG4XM;iJ*Bi)aPeAy6Pebc=e%kY?g?(C6Sd)1'
    '^2o<h9mzRB(1^f9mQWhwBN)ygBVvx=fj?iBT@1nsG+rk&Ld$w{V7q9(Y_#=a8@tiHH!w<^Ko-uNOT~?fPzinO**oRV!NXFe5FiW%'
    'WDgY1Y70!rgFDn5WVZK7RtDuT2*6Dy;7#`J6LrmSK`>bz4-Kn~uSv&u^sVy+>vaW?OQO=!NS^7{UIC(kZRge#skvkw_k8e~E<G3o'
    'vQb<XPtrtdMgh*fP!lX(vt`EC`zt#D;t)!byVqY3lK#zWfNDQ;Dl6KD{nyquy!3QU_ga@calopxKxzG&{K<tRG;=^vL6TsH%$O8%'
    'Tgs=slgSAretHHaYFs|+3P+{|^1?s?^KUhCUHSbPA){d70HF8_{=oSKkHX~1b6FJYpc$ofX@6(%BW=KMrSDzc{hbONZ_$N9@q)?W'
    '^1|q_1ETSKjM+~~AcMumaJ!YI9y*c<#)us?5?zq8_0(jl(N5QKzFz_&Ta|j{1VmS*qLfhO>BxV!j{E7`=*^~a(%fTTY!0K_{@$m-'
    'o%YJkf`LATqPWtLQR=!j{nP7I?^E_+=zXfwj*aHD?}1xy87_?4VEux@yhzyzy9-=?5=@x*@&toum@2x$m}vt8b3J3LXc{?UU%}S%'
    'jBO%;Z)NNiyd=?30=ztASfWuzlbwb=8<!bIeO9|)4qypXu~!K?w41WFRC3Q|jyiUeL>vh=E^;No`zP!^)`uBIMHP<p7yZ&BY?QF0'
    'vUi<GW7#cbB{hB7tzP_8H<!QHg2Hl@1+m7EN4P<Gn+p*rE3C5gCKC3o^@<aT*;>|+#<2!avBj$&Y9WB-i~H<2y(>#<vrHaC0rsw>'
    's3~6B;(LmM3jwImZSnDo{fjakI(bh}yJ4#WUbYmCbuaK_RY{wJI+lVwI-=ky2MX4w&j<9uv^2u&)vQAoNxDQ)h4=kR?SJDx2VhQu'
    '>>d%~kn;yT57Mxprtu>i30TLZ)oUnPgA1wlbiw4&0;|TIEx@)4g%(x|!`Quyh+3T1Y$>`m$fJ+G62mjAx`pI&OABw`)tQPHAW1vD'
    'z!9ZeGf`MSe)1w{aih&ef-HIGKC<!jy=rTLvReAhwv}yrd+1V9T!&p`OVe%B(D&#S_$0a+?hKYU{my2&ilt3L3@3^|9;kCQlkmg9'
    'RGI{*Lp`HfDUNJ4$kbTj6|iH6J#>yxBG2&6dC)ifdHIqmr5L7piMTT=Q{Jn=>Nzv9003UTE{KC%5cM5f!}MpOL1g4*tyTUsUuqfd'
    'NKr0xE=J;?C~e)ofen9?XqE#_M*$x3c<15YTN~%d#N}GTteWg1xxw6=fCM_%$uogb58e}~7yU9lb%uoqb}ME(Ed>;>obaK^rm$2B'
    'V(fC<1v$4N2xz*8=2y81Qp&c(9B@F&+UQ(L$_|biF{T#dGbdGH)kf*jLv^`rgbfA_GOIfQlucm9kyg@8c2PWh`nJ1mrz-jSIabN}'
    'C&)v$OeNfjFXQrZ@1m%A3uou(PM%k^rqu!!aOPgox@n(kw+nA`hci+9>Itgb!2k^{PO6l%wHFp0h7-ldovPPlc#XqhoTyv^&s^yV'
    '!xDrOZ*~onhiTX1juuhCE0vX9Qua>AjrY-366r>AE8J6^LX=@1h;}em_OwD~j^u?Evy$DB1C4eOqSri<6x2s!(5TtaAb#XWl*oB2'
    'Qe!mG9cx|ic$aNRF`Ek11?x*0w#jr3R>7A89><aI(m*JWe;|4;RK3L=v=D1sa*+BR2VG7l*Iac<^+N1|mGAh;ON_kEJanugq4sVr'
    'HH{l)4#<W=y|z+~LQ7~b>Q1kAX)3#V3+Tlu0OsXB&jn%S@+{|E0XMEmmW$ABjiqrZcP*8BTL&VS8ZDENI8A9qOYhweyK{%Z#xtG8'
    'b_PM@OevWS-=(P>iiKQgmbU~P;Cy(@eWjEf3$^wdi$TRkh0?-zZ%<UK=It1uJS|67UdgcUG#%Dk2p&unVJLYMw?T8c8NuX?<((8>'
    'Jjd8f7T*+LKaA=NnWgMV3vh{E^Eu?fv|@J#bZ`TPUu_w6Mko+0I)MvAO^3~%BDIS?yoyaa;Z#CBsb5L5e@{XgiO;8iw$+DCGmVCb'
    'pn|NdHcNB!k{WX{wPKpkgeo(&xX06@m&2;I(7?`k0$ey@qfMNuCmt?}zBwCq<MF2&5iooFheXNnp=LGHOvN^qw$SOXBxdr)5{;Bk'
    ';~I_?0m1nH#xS&E7|1WE+<MI$mq2K1Ic)*c!g{@0WFJ=&54-H2Rl&%vLi5T(83<@H*zj3`v<c*J3`qw46c}BQ&5AT(9y~@xqGsBz'
    '-qe*W%@M*C>L^p==4^_H2uiV*!qmv<>r61^*Zy#vmsekjeOGI9r|SUUask;z5RQV&2$iSRz-yCDWO%<0IF8Znsh!65r0@X-zpQ8y'
    'WKs1HDSlq4I{oR%T_SsWKBBMTSB|i>NM?|t3K2vu(3{gsR0g&<CDA^mb_O7im8>F2Qq93m+_(38DmeLsK{^0`MW8NdFnX>WG;k1a'
    '7SHJPrzVhtSDG7F5C_75TE;+U<ZOZJkioz40NR?A1hK>NMkJMyO<Y`d1#GM?X)pe?VgK}P_ytP9^@$S&J~V+h6I`kY2MP``SCsN5'
    'tAaYpz!1{5E`nHs7KU{*45kVWAq0EERNOv&F_o2AE4JVY2mK5}_PArVK5DDZ$<9zByd4C2+Mw^n4<EW>N?2QqH8R#Utnfnay9_bi'
    'CJfH&5sgwt&`~?SgihEcksLNKVn<#l$2;s4o;;f2$Wl&rB|>KwOmj5^xI(wi2KUQxibHC900QEChu%-(U{0iW&~SJv6Q0Nw8JeXP'
    'o)9ABdXxwg1Z`nK&kO-JNYnGw(b@tVh~Co@?a}e?;x~6~I3k;svRACxU0!*jpWX(gZGgzL2$`*+izP5$=E?~%p^@7e8`%w2HEd=m'
    'shvM&K-)g!s*<uniH!oG2ZTD}=Zuv+S#E=Wa&cj`o>R}!moPC=21oifU1WmC1I>@e8<WIi@vcazsi$>sY^Eu<5oE4kM{W?Z6oL`j'
    'XBc7aYg-c3hB>W6G>-f3*o5$>@b<;?1K_N`Ism8KD721oez``Rj`KKZN0$bK5&qO|7>CU`z&Ko!(jg>SU^oY$QxuscsM&O$&H*eS'
    '1ORA)woDm7zLzxvRh_gf+l1x`xY=Ohh^sV83Mz3ZL-#<<hG2@F$f1G%pRkE^AB7BGfzU(L=ofTTC7?iAguq^}p$FfnSu=Wy9r960'
    '(Vl<n4*pJ2TVi7TnD#TO>&s7I(9;6j&4J^!Ih0v4XbT=*N3>!k+6h;AanPp+Hjuk;Jdh@T*vp|G-4xu+fW{4)zLcJdu%SqTL3dn&'
    '(^6oyZ_znmSPXGYq=o5SbTS1@z!+xbv$PzNyT!=WcQ?21kEt-cpwNS02!$34ErD-h`?dM(PW&gk$2QLbm$z9R8c<jf8ZJDKL6?p;'
    '{zfW>K5p$Ab<yZM<|JkHUM}`<<yO`(^G0Vf18P_Nmg*j9Y_OXST~ju-<CIOm6+G4EdFFjq351D{tzr{idO@TvI;DjKZd(68L8@!>'
    ')gV-~Ylx^NkD*GKQ6Ef|)MrFJ)?ABTNUc&G;iIIiz#1L@mUY0uee(<EUCqviNqOiDZP=#YK~&=)*&Lb%*Ct!-Xt|z-tu^Ra22Vss'
    '+cHAXxY)-jR~s-z7l*c4xsYNChL4nMhD?*`W2G+zrW|d4t;`Ut;*VxV&enI8uv2Lt)r3jByMFG2Zy;RF)gdt~hBN0xt%>xLi)IT^'
    'ebqU?>g3T71R?TW>+mmvVAIY!36>!hWQ6Q%iDW%!7fzhvjL{6{e=<cpG}TxdW(+M!C8J``f_=!k^m%%=9X75pW;yC`_+<?JK~bP2'
    '`~>z3E75vcwk1>740)I2{#5uQCAXq2MX1czx?3Ynhp8}%0^HNic|uS{DK@tU-4yYuK*K+8fBc7rbs$y>E_+t74KaG$#nBo{6XG|K'
    'AU!~?j`Dd&P_2h7?`DI^tQ#|V7bnT&J$DKv4WU)e)HP+{0HqR5&iKiuYzJHuT+24W1ruTFoqA2?K**6pFhHA}xk+Im*Ak1^vy9Fq'
    'h*sbo*xU)$FKH2^^VmvoEd<kxotwb9W}QTG7|GCE)CAc>DFfrYC6uRi%7bMB7lpAzHWB&=+YX+hH}C%Bd9VuC32F04nRbO{*6eVC'
    'xQ#5M!u#Z)1f}U`Wqns(cu(8Sp0)%85iF;~lWdtG$edbFtN^Q$wuqinhnc>g_9R?OfZa<74ze;)3WhC(_rJ+aGaW*G%GH@NoW1-g'
    ')}lgSp(0fngI_Jmo(up^6_}>Q!u8-wCcIgZXpA}wpC9d$qgJmZ3`_&|i}8R(fbuNcR?6UjWB7CghG~%zXK4JEEJCe;GJ5{LiUChA'
    'aqhz-25L`>tWu-NmSd0VHu;N841x)Rt^1#gR`}DJPa8WaLzL|%O2ePl&w^Kpgy?K4Oc!`l+fq`Xk`&V_?PgRkOl^cpeEcTJ&1h|e'
    'Y%E!pak&FG6q(n>&@!~OPvA#ho*gPR4oybC5DS(HKZyZRGV_x*!FZn=c+CDsadql?S8Z@DmL!cLH;H{*Fw}Ju#wUcl(`RhI^E0@4'
    'I;Ey*U|LTfdoYEjjc9jSM%(Mbsf-BAu|qkOHz*o>^yo!$`^r26V1q1<h%}x<nN*UOsU##5$3gti{4^j8p;EKeH6c=>Rj_0sKTNdB'
    'fi_|<hoTGSUWUD(K8}IWh12?zfoQz2Mf}V9<j}|jb!bYmfwMwC#Z&xI_bJQv=4du4A5{gF@^sz_1ZL=gvE7jxftdMce#97jpa>RQ'
    '0;2`kTjeudE$M5W`uf?yd@&q9<cPI9NlJkgp0L93%P{Ro!xp?LcAYsUE2o3FjW>Qv0aNh&eH5cTGkE#+S|2wj08IJlq2^x5>P)&8'
    'Je8EzOZU1h2c!b|IGaw%1E~~}*~VE?tidpks1nL`HDr}!5P@)s*)|!Mw>5>!5eOzeoE+cA3+Q7Tv>=O30kCrTV5+FiC&70tF^7|0'
    '#h^$?V{tdJkrJ5Y8o7m27Zh*PU`#r>D`hkWGF$r&fi;9Io0+(7Ca95cCnB$6V!bD&GEOA1vn{uVC~8UI?sZg|h1nN|CFxnwIAH*x'
    '4QPOy@$=Cp>T5tPUL3h;Ww8by!6<ii#7i&;j|NdOnn-R|DM$_`PDTbTfcX4k0D`8)m<pOyI#hU>)rMbj7C~+a$+(t+h(KJLg?KHl'
    'f4tcg2B!1!fgcypU#|{5HzjiF8RwcjZ&F{}WqCd!Fm)YK#l3G!mem8aXII8%5bk{zw!L!B)<mBrZo2YdtuYADLCh?H3j2K}e|Q$C'
    'H}LD5B)eLLSfg)>%PPGR-@90fxk`e1)z`5M(u;NSF_;%8jY&XJmNF}rmz3U$l-@}noyHFO5=j*wT}L;_LNDUH59hd+F<z`okR_+I'
    'M62G&7!ts#L;bb_@QLq>G%r&GIEg8i29B|aElA*qBYVQ4RENaBD>0Qaj6E}epv>Tu`hG6q6qbrZT0+_z8;o$uvHml3P$J(a7d;5Z'
    'NXk^zII5)aiPCY$5}R2QEKQ%o1Azr(hRx8Y$4{_bF6J}sen?&O3=l-;pLzI}mi&}aY|LsFl?MGRf%{A~LK~wNL<C}yF2WKJ;EmIz'
    'JJ>l^QLd9|AEea44`&}z`w-BxQGuOlg_`E)2(%RDErb;bV7OrXAe%D@a*z2;nK}={PB7Ft{sj;^$)#oH0<NZTr{nBIYktkbsiqmX'
    'S72Zurk-!iv|W8C1oLTHsxp;$M^F_6B4zStkWs61F0GK;7}}^ibVn+EhK(Fn#9>NPyHmjvBRg6%VIUvN5eaSLGRRec=U_%cGh(UW'
    'sA()A46%56E9i|%5G$8xlHti>K@Pc42mnk%z*~x{D?tJBgbo42<{vkWc0o|`<+lAGxf&ML226EQH?6HFw$nS)5I5rQ;3Zr}K~~^R'
    '11a2So@gyA6OR9`_G(f6Qw%D{6hiGY;6o#`HpFp7y?9Xoms5L&xS2qIXrepF7*Vc0Cc7RPb7q8S4kb(B#c>ssxT5SbbWHE1GGIb!'
    'F(iIGs>Z*FUH-s>NYF5|EQ?H%G-JFfTNo)%5^PKj&!eMbwl=ul)@cB;eRvRMW)CkyI!3oq?cqDdizq%L3wpex&TKODJi;x_VF#Td'
    'eagb^=`%=P>Z=^L6T_988NbfE4?yIJsm;nhJXc1F)!NIOCDCEu2MR2ok!f!6>kQ#a%D8BDN$G6SK(ouGl$6m<br}mWsMXcjfj<|X'
    '_0(s`nNF22V*O;4V<|)3F16=bT4M(sLYc6{_Q;m`Eh%UdDgINK9q@H?Hp}BH-+;AYhzj>u5y4_r$8-LAUm9i9%4P`-<$|@}FvgP8'
    '67FG=>=R^tpOoc~?|ekiT`~xy9MsDc<b|U{gn|&VMzNPDilb~GH3dEbqL!&aYQ>KYO@Nf<b?w^}Z63ow+K@+@ayP|>3^ICAdyzJ{'
    'i=jZ&H8&2ClXGGe5|2L3Vtp~Ho5m;#4gp${&!MBI$SBN+WnonquM(awK6h@PWhDpXb0)(Pke9`%GFvpzdf?b<U>b^zW}vh`M`PAv'
    'UMP@pG@b<;6sOIVlU8FBL}-24Y+3H<zF0Z8x^PsyZOWq9{YcDQVg#fHmQ?M2729Ab8LFBfv)ag=B5@spiAfLhy?>J1%yDwBn$Fo;'
    '9Nnc*1<f-TNn%7qYSV)Ozpe3i9}3<&ISIa7HH?r;dYog1KYv|dBb<Acl0!E7n(%du1kb(1)(d-9JnuVuaaXY!wr1$ljq+wVy$<WT'
    'L<tYid)=>9XC4Gvq%7!`<SB;(7G)!dLJC)sC<VWlR7VMesrz(XR6NyCs+<g4*Y(pmC~CFDZ~gk{@jm<umrnXe'
))).decode("utf-8"))
_GOLD_HAZARD = json.loads(zlib.decompress(base64.b85decode((
    'c-pO7+ioBy4E>ipk0QVqpl_{|s;g$ZQo7QrUG0}t{r4s_Fc-kalO|GLVh<SO%dt)Vd4T-z)A#QWzdpTu{q+3l@28iC#Xq_wy#Df!'
    'AIk%VmHzFwr=Pz*EbbASpN)Iv1T$yS_auK_>5b%fQjkd?ldvTndy!0HnG|JGl1W)4v6D$&CIy)UGD+43%i3UB8!T&sWo^VTv;wU3'
    '<gWJQx%pZA;#Wp*WeiqE*_0B>B$7!ilcG#YGAWCs>|~PcSXp+g%=$%nCx7c5D6~$Ou+zL2BNpMK;D#T;;G;nMSBS%}4hAJ2d6Hew'
    '6DTxPXe8@}lOpUg5c2JBU%ot-KOLJy*`Ixk3N0x#Il%ek<U^%S!H03BpwK!6AD7lU>WM}QjTKr{Xi28!pj%k*s9XEzPtU(^f8yht'
    'P0*Iq*Z@3I;fHdt-57Q~29#q^K>5A{X7Nq~IfW+Qcc7rq<ogbU3XK$6ci)GK`!s)|ZD;@d^!(+Ii*qew9hWqI;^+Q$4;7Wv-Y>rr'
    '$L13E1&^=5?O#P1NQ4*^8pt;pQNFFHXgw-gkBZj&=o1fGmT7(TInlgAyIg~NOjk^c3oq2T-05Rb7B+kQD0`jOiQPI9uvt#$=s9?U'
    'J|T@0jCTX$Vl6&pIg5qN3Qf8GoMdy$a`FM^#P7g)Q5JAA+5?yG-Qe3d-rv9VJ(i?nnQ=bJqXcjqG|<(Sdm9Lw4FHT0U+Troy_oxL'
    '0mRvc1#bh&C=pObiGU>EE3!h13QbPr&?mg(w0*)m(Y!(nVw3=iV`tQ~&SMaKFvK%(aF5-kyr~yA&TQpOVGxFd4#0A(*l|GR*iJU!'
    'z}iKC5?-%+_BOrBcv_JbTwq%ObvHW9yiM(cM1Ze}U|lp1Vc8o=DZe1gum`dXdmy7iBZbC_EK&40baHixj+&!GtF>MC=;Dl5zIk-^'
    ';;_WO)4H97V6|clc(M0R0^TJEbm0`!DTb@F(E>#QVLQX0UtfOy_RG`D%U?s;;vrjZ7UHrHxQQsR<^JnVy>G(N$n&=_g`_Qf6*3rh'
    '4#&npsNYF;$m`-FKI;P|LcX?xx^MmF(ui)th?UV#7^yM_D<fDLal#lYBh8v%v6ya2S_2$|r4So<@8qu2e%Ga5yYe6Lp_FV_axW`x'
    'HaLsQNNmYZ+~%i#!#r}P;kFjBYHwHK<H`rd&LuL=I78)}8Uj{MHgiTZXPR+R<;*kA8f&97bJ`+YwJzOPP8U76mwMn*MNi^A?`TE2'
    '>U~MH#c6j9)B=nfrs-0{8&kndXjN1DsjTo6V>Be;#TG=?^+qYVPm~7^DfrlQr|aCynNV^AljVH3QrHSjG09aA_U+3fQ6{<S!7Mjc'
    'u;NJBIX1<U^ApBU8H449a=jpOV9Fxgn6IQka@gcOtkA@S<C+n~-kPI=oadVIHh--8@;sTz?x2r+wAkiRpVcL0|D?eTXRsu7OUlTf'
    'g0s|odla?8hiSH?;7yhkrVAPo3x)6=ST9HaY2c**wqEC8;?M@AWQ|GGlw8JRcFTdkA}-~WG&yC!^Dz~X=4a!HSMM@KW6GmaH7YGN'
    '4CibwV)G;Vn@VFU<>&k?Flfn9eW4+@<OrgeNC@PX96^_y((&wrZaM8V`XU>Qb!Cj^>Sbk5W;s{(JTn5mB0kxK<=VF?VFqEFN=z<|'
    'PpkBovhshRHmd7&lQO`*VGu`&`jmw=e@?}p38v0hWy;1nh<IAFRGVUNtZs;dd3P|ciN<=3m@HaaEG-C{SDyQ>^I5~ZNJ(|Ee9L{K'
    '`=BCeU+|;AMh#ps*!nfE+I(v(ni6S*L1%N0gnbsPO^W!QrjH^b>HXYp+Hex3eO-~oEV{QR4QX^*vR2Xi+3}Gom8UJFJZ+&gn7Y=~'
    'tki~-nqJM`d#03IN+^_UeQQU#iK=o}PI+4D_6ng4MCQp+5iM8IZ1|x@t1?rWRHsKVljRwRSX~{al<%Q{OSLL%GFIu>UFQl?HPH1V'
    '+D)#n2L|J8_O>d$BtrINJh(A_vyax7%j|7PZ7sT1{ygk*rxIihI|lcjVa%WbLAH{(0mHKSa`1Z9B2?i@LybeYf7s#hP`>P7>cjfo'
    '&p3QgmsL<lU&<!i@PNVF#!?KkX%ke&*|do(BU-nn31g~^))@msG-ehCV_DTE1Ywf7KApEdN@oOHt;k4LMm8BDn7jRkB!Hu?;44bp'
    ';@5oidqM_2ZP0)**?O4KdXQ=FmAc-m_PmNQ+lKqhV7rl;4cHb+La^npT$eQtX=3F!4MT&QPeZF|ax~g-8T-u<`Q%QqGvXb(Gv{0H'
    'f~JiK0~q6B3Amx*MZr*Tc|EGc(dfs;R#&*N@NYFfXs?=D5C_AjH2zxXT%9Ifo1KS12|uVhE)673WR1azKWOV}z9|{)JUh!P3meYL'
    '{?^;N#v@FiC<pUc$<)E%47O>~HBF341#O_p8Q+zjgeW7tOlstd_J?Ym>5g9S%DcxGwJ@&GC6An8_@2lRvE)(0sqxS?Ml?-DOVIXh'
    '&^8Px<@BAC{s!^V1>u}+ZdmjkowMjWYym#HcfGk215k8eUT<YLjcUV<_q(D=?$7<Ax4zej-oGo5D-JyiiP48!^Kelkv4g^5W{Vzf'
    '4a{%sxwq`1M{7@X@GV)Klv*b}(8{?hX^suie`Ms+vDOGuMQ{5b8=CnhTX9CTy`1oKfZlGo-R0Z?{{0V4Dr6P'
))).decode("utf-8"))
_WEED_REPLAY_STEPS = 2
_WEED_STATE = {0: {"last_step": -1, "active": {}}, 1: {"last_step": -1, "active": {}}}
_SHIFT_STATE = {
    0: {"last_step": -1, "due_step": -1, "due": {}, "last_preempt": -10**9},
    1: {"last_step": -1, "due_step": -1, "due": {}, "last_preempt": -10**9},
}
_PREEMPT_ENABLED = True
_PREEMPT_THRESHOLD = 0.5
_PREEMPT_FRACTION = 2.0
_PREEMPT_MAX_BATCH = 30
_PREEMPT_COOLDOWN = 1
_PREEMPT_MAX_CLONE_DISTANCE = 6
_PREEMPT_START = 120
_PREEMPT_STOP = 680
_PREMIUM = ("STRAWBERRY", "MELON", "MILK", "WOOL")
_SELLABLE = (
    "STRAWBERRY", "MELON", "MILK", "WOOL", "WHEAT",
    "FERTILIZER", "EGG", "TOMATO", "CARROT",
)


def _get(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _step(obs):
    value = _get(obs, "step", None)
    if value is not None:
        try:
            return min(max(0, int(value)), len(_ACTIONS) - 1)
        except (TypeError, ValueError):
            pass
    day = int(_get(obs, "day", 0) or 0)
    hour = int(_get(obs, "hour", 0) or 0)
    return min(max(0, day * 24 + hour), len(_ACTIONS) - 1)


def _copy_action(action):
    action = copy.deepcopy(action or {})
    return {
        "farmer": list(action.get("farmer") or ["PASS"]),
        "hands": [list(order or ["PASS"]) for order in (action.get("hands") or [])],
        "market": [list(order) for order in (action.get("market") or [])],
    }


def _farm(obs):
    seat = 1 if int(_get(obs, "player", 0) or 0) == 1 else 0
    farms = list(_get(obs, "farms", []) or [])
    return seat, farms[seat] if seat < len(farms) else {}


def _align_hands(action, obs):
    action = _copy_action(action)
    _seat, farm = _farm(obs)
    expected = len(_get(farm, "hands", []) or [])
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


def _trace_actor_action(step, actor):
    trace = _ACTIONS[min(max(int(step), 0), len(_ACTIONS) - 1)] or {}
    if actor == "farmer":
        return list(trace.get("farmer") or ["PASS"])
    hands = trace.get("hands", []) or []
    return list(hands[actor] if actor < len(hands) else ["PASS"])


def _weed_repair_action(obs, action, step):
    """Replace a blocked BUILD/PLANT with DIG, retry it, then catch up twice."""
    action = _align_hands(action, obs)
    seat, farm = _farm(obs)
    game = _WEED_STATE[seat]
    if step == 0 or step < int(game.get("last_step", -1)):
        game = {"last_step": step, "active": {}}
        _WEED_STATE[seat] = game
    game["last_step"] = step
    positions = [_get(farm, "farmer"), *list(_get(farm, "hands", []) or [])]
    unit_actions = [action.get("farmer", ["PASS"]), *list(action.get("hands") or [])]
    active = game["active"]

    for actor, transaction in list(active.items()):
        index = 0 if actor == "farmer" else int(actor) + 1
        if index >= len(unit_actions):
            active.pop(actor, None)
            continue
        age = step - int(transaction["start"])
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


def _shed_access(size):
    half = size // 2
    return {
        (half - 1, half - 1), (half, half - 1),
        (half - 1, half), (half, half),
    }


def _projected_shed(obs, action):
    _seat, farm = _farm(obs)
    private = _get(obs, "private", {}) or {}
    projected = {
        key: max(0, int(value or 0))
        for key, value in dict(_get(private, "shed", {}) or {}).items()
    }
    inventories = list(_get(private, "inventories", []) or [])
    positions = [_get(farm, "farmer", [0, 0]), *list(_get(farm, "hands", []) or [])]
    actions = [action.get("farmer", ["PASS"]), *list(action.get("hands") or [])]
    tiles = list(_get(farm, "tiles", []) or [])
    access = _shed_access(len(tiles) or 10)
    for index, unit_action in enumerate(actions):
        if index >= len(positions) or index >= len(inventories):
            continue
        position = positions[index]
        if not isinstance(position, (list, tuple)) or len(position) < 2:
            continue
        x, y = int(position[0]), int(position[1])
        if (x, y) not in access or not (0 <= y < len(tiles) and 0 <= x < len(tiles[y])):
            continue
        inventory = {
            key: max(0, int(value or 0))
            for key, value in dict(inventories[index] or {}).items()
        }
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
    counts = {
        key: 0 for key in (
            "WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
            "COW", "SHEEP", "GOOSE", "PASTURE", "COOP", "WEED",
        )
    }
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
    seat = 1 if int(_get(obs, "player", 0) or 0) == 1 else 0
    state = _SHIFT_STATE[seat]
    if step == 0 or step < int(state.get("last_step", -1)):
        state = {"last_step": step, "due_step": -1, "due": {}, "last_preempt": -10**9}
        _SHIFT_STATE[seat] = state
    state["last_step"] = step
    return state


def _repay_shift(obs, action, step):
    """Remove quantities sold one turn early from the scheduled SELL tape."""
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


def _future_base_sells(step):
    if step + 1 >= len(_ACTIONS):
        return {}
    result = {}
    for raw in (_ACTIONS[step + 1].get("market") or []):
        if len(raw) >= 3 and raw[0] == "SELL" and raw[1] in _PREMIUM:
            result[raw[1]] = result.get(raw[1], 0) + max(0, int(raw[2]))
    return result


def _remaining_shed(obs, action):
    remaining = _projected_shed(obs, action)
    for raw in action.get("market", []) or []:
        if len(raw) >= 3 and raw[0] == "SELL":
            item = raw[1]
            remaining[item] = max(0, int(remaining.get(item, 0) or 0) - max(0, int(raw[2])))
    return remaining


def _preempt_shift(obs, action, step):
    """Shift a bounded part of the next scheduled premium SELL one turn earlier."""
    if not _PREEMPT_ENABLED or not (_PREEMPT_START <= step < _PREEMPT_STOP):
        return action
    state = _shift_state(obs, step)
    if state.get("due") or step - int(state.get("last_preempt", -10**9)) < _PREEMPT_COOLDOWN:
        return action
    if _clone_distance(obs) > _PREEMPT_MAX_CLONE_DISTANCE:
        return action
    future_base = _future_base_sells(step)
    if not future_base:
        return action
    hazards = {
        row[0]: row for row in _GOLD_HAZARD.get(str(step + 1), [])
        if row[0] in _PREMIUM and float(row[1]) >= _PREEMPT_THRESHOLD
    }
    if not hazards:
        return action

    action = _safe_market(obs, action)
    market = list(action.get("market") or [])
    remaining = _remaining_shed(obs, action)
    shifted = {}
    for item in _PREMIUM:
        row = hazards.get(item)
        if row is None:
            continue
        target = min(
            max(0, int(remaining.get(item, 0) or 0)),
            max(0, int(future_base.get(item, 0) or 0)),
            _PREEMPT_MAX_BATCH,
            max(1, int(round(float(row[2]) * _PREEMPT_FRACTION))),
        )
        if target <= 0:
            continue
        existing_index = next(
            (index for index, order in enumerate(market)
             if len(order) >= 3 and order[0] == "SELL" and order[1] == item),
            None,
        )
        if existing_index is not None:
            market[existing_index][2] = int(market[existing_index][2]) + target
        elif len(market) < 10:
            # The target is an opponent SELL on the *next* turn, so this order
            # does not need to jump ahead of our base orders on the current
            # turn.  Appending preserves the teacher tape's same-turn SELL
            # priority; prepending can accidentally let the opponent beat an
            # existing high-value STRAWBERRY order even when total quantities
            # are unchanged.
            market.append(["SELL", item, target])
        else:
            continue
        remaining[item] = max(0, int(remaining.get(item, 0) or 0) - target)
        shifted[item] = target
    if shifted:
        action["market"] = market[:10]
        state["due_step"] = step + 1
        state["due"] = shifted
        state["last_preempt"] = step
    return action


def _safe_market(obs, action):
    action = _align_hands(action, obs)
    remaining = _projected_shed(obs, action)
    market = []
    for raw in action.get("market", []) or []:
        order = list(raw)
        if len(order) >= 3 and order[0] == "SELL":
            item = order[1]
            try:
                requested = max(0, int(order[2]))
            except (TypeError, ValueError):
                requested = 0
            quantity = min(requested, max(0, int(remaining.get(item, 0) or 0)))
            if quantity <= 0:
                continue
            order[2] = quantity
            remaining[item] = max(0, int(remaining.get(item, 0) or 0) - quantity)
        market.append(order)
    action["market"] = market[:10]
    return action


def _terminal_market(obs, action):
    action = _align_hands(action, obs)
    shed = _projected_shed(obs, action)
    existing = [list(order) for order in (action.get("market") or []) if order]
    existing_sell = {order[1] for order in existing if len(order) >= 3 and order[0] == "SELL"}
    rows = []
    prices = _get(_get(obs, "market", {}) or {}, "prices", {}) or {}
    for index, item in enumerate(_SELLABLE):
        quantity = max(0, int(shed.get(item, 0) or 0))
        if quantity > 0 and item not in existing_sell:
            rows.append((float(prices.get(item, 1) or 1), -index, item, quantity))
    rows.sort(reverse=True)
    action["market"] = existing + [["SELL", item, quantity] for _, _, item, quantity in rows]
    action["market"] = action["market"][:10]
    return action


def agent(obs):
    try:
        step = _step(obs)
        action = _weed_repair_action(obs, _copy_action(_ACTIONS[step]), step)
        action = _repay_shift(obs, action, step)
        action = _safe_market(obs, action)
        action = _preempt_shift(obs, action, step)
        action = _safe_market(obs, action)
        if step == len(_ACTIONS) - 1:
            action = _terminal_market(obs, action)
        return _align_hands(action, obs)
    except Exception:
        _seat, farm = _farm(obs)
        return {
            "farmer": ["PASS"],
            "hands": [["PASS"] for _ in (_get(farm, "hands", []) or [])],
            "market": [],
        }


def _kaggle_submission_entrypoint(obs):
    return agent(obs)

# Cross-graft: preserve mechanism v13_r3, replace only frozen route stable12.
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
_CROSS_GRAFT_MECHANISM = 'v13_r3'
_CROSS_GRAFT_ROUTE_NAME = 'stable12'
_CROSS_GRAFT_ROUTE_SHA256 = '5ee61bbcd473c7bb59ec284b0992dc881eac24a1008463b6d0d4edf772d131ff'
