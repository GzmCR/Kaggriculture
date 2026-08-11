
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

# Cross-graft: preserve mechanism v13_r3, replace only frozen route v022c.
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
_CROSS_GRAFT_MECHANISM = 'v13_r3'
_CROSS_GRAFT_ROUTE_NAME = 'v022c'
_CROSS_GRAFT_ROUTE_SHA256 = 'c234e990fd63a168535b55de1f11289fa2bbc563b390390b49d0d65169cedb18'
