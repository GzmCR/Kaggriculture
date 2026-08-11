
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

# Cross-graft: preserve mechanism v13_r3, replace only frozen route v22.
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
_CROSS_GRAFT_MECHANISM = 'v13_r3'
_CROSS_GRAFT_ROUTE_NAME = 'v22'
_CROSS_GRAFT_ROUTE_SHA256 = 'cd4380d55c4a13c2ed4fd0c9463268c5764599f7e3f58b91e960b49d7dfd5d77'
