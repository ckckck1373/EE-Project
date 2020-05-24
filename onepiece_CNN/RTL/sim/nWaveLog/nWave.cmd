wvResizeWindow -win $_nWave1 1920 23 1920 1017
wvSetPosition -win $_nWave1 {("G1" 0)}
wvOpenFile -win $_nWave1 {/home/u106/u106011206/OnepieceSR/sim/One.fsdb}
wvSetCursor -win $_nWave1 217986.538186
wvGetSignalOpen -win $_nWave1
wvGetSignalClose -win $_nWave1
wvSetCursor -win $_nWave1 236719.756312
wvGetSignalOpen -win $_nWave1
wvGetSignalClose -win $_nWave1
wvSetCursor -win $_nWave1 448319.970137
wvSetCursor -win $_nWave1 54496.634547
wvSelectGroup -win $_nWave1 {G1}
wvGetSignalOpen -win $_nWave1
wvGetSignalClose -win $_nWave1
wvGetSignalOpen -win $_nWave1
wvGetSignalClose -win $_nWave1
wvGetSignalOpen -win $_nWave1
wvGetSignalClose -win $_nWave1
wvAddAllSignals -win $_nWave1
wvGetSignalOpen -win $_nWave1
wvGetSignalClose -win $_nWave1
wvDisplayGridCount -win $_nWave1 -off
wvGetSignalClose -win $_nWave1
wvReloadFile -win $_nWave1
wvSelectGroup -win $_nWave1 {G1}
wvGetSignalOpen -win $_nWave1
wvGetSignalSetScope -win $_nWave1 "/test_top/top"
wvGetSignalSetScope -win $_nWave1 "/test_top/top/sum_RD"
wvSetPosition -win $_nWave1 {("G1" 3)}
wvSetPosition -win $_nWave1 {("G1" 3)}
wvAddSignal -win $_nWave1 -clear
wvAddSignal -win $_nWave1 -group {"G1" \
{/test_top/top/sum_RD/sum1\[42:0\]} \
{/test_top/top/sum_RD/sum2\[42:0\]} \
{/test_top/top/sum_RD/sum_all\[43:0\]} \
}
wvAddSignal -win $_nWave1 -group {"G2" \
}
wvSelectSignal -win $_nWave1 {( "G1" 1 2 3 )} 
wvSetPosition -win $_nWave1 {("G1" 3)}
wvGetSignalClose -win $_nWave1
wvZoomIn -win $_nWave1
wvZoomIn -win $_nWave1
wvZoomIn -win $_nWave1
wvZoomIn -win $_nWave1
wvZoomIn -win $_nWave1
wvZoomIn -win $_nWave1
wvZoomIn -win $_nWave1
wvZoomIn -win $_nWave1
wvZoomIn -win $_nWave1
wvSelectGroup -win $_nWave1 {G2}
wvSelectSignal -win $_nWave1 {( "G1" 1 )} 
wvSelectSignal -win $_nWave1 {( "G1" 1 2 3 )} 
wvSelectSignal -win $_nWave1 {( "G1" 1 2 3 )} 
wvSetRadix -win $_nWave1 -format UDec
wvSetCursor -win $_nWave1 349505.321270 -snap {("G2" 0)}
wvResizeWindow -win $_nWave1 0 23 1536 801
wvSetCursor -win $_nWave1 349625.016153 -snap {("G2" 0)}
wvSetCursor -win $_nWave1 349288.914668 -snap {("G2" 0)}
wvSetCursor -win $_nWave1 349724.262001 -snap {("G2" 0)}
wvSetCursor -win $_nWave1 349722.594003 -snap {("G2" 0)}
wvShowFilterTextField -win $_nWave1 -on
wvSetCursor -win $_nWave1 349172.988846
wvSetCursor -win $_nWave1 349473.228386 -snap {("G2" 0)}
wvSetOptions -win $_nWave1 -snap off
wvGetSignalOpen -win $_nWave1
wvGetSignalSetScope -win $_nWave1 "/test_top"
wvGetSignalSetScope -win $_nWave1 "/test_top/top"
wvGetSignalSetScope -win $_nWave1 "/test_top/top/sum_RD"
wvSwitchDisplayAttr -win $_nWave1
wvSetCursor -win $_nWave1 349345.626581
wvSetPosition -win $_nWave1 {("G1" 3)}
wvSetPosition -win $_nWave1 {("G1" 6)}
wvSetPosition -win $_nWave1 {("G1" 6)}
wvAddSignal -win $_nWave1 -clear
wvAddSignal -win $_nWave1 -group {"G1" \
{/test_top/top/sum_RD/sum1\[42:0\]} -color ID_RED5 \
{/test_top/top/sum_RD/sum2\[42:0\]} -color ID_RED5 \
{/test_top/top/sum_RD/sum_all\[43:0\]} -color ID_RED5 \
{/test_top/top/sum_RD/pixel_out\[15:0\]} \
{/test_top/top/sum_RD/temp_pixel_out\[15:0\]} \
{/test_top/top/sum_RD/temp_sum1\[42:0\]} \
}
wvAddSignal -win $_nWave1 -group {"G2" \
}
wvSelectSignal -win $_nWave1 {( "G1" 4 5 6 )} 
wvSetPosition -win $_nWave1 {("G1" 6)}
wvSetPosition -win $_nWave1 {("G1" 3)}
wvSetPosition -win $_nWave1 {("G1" 6)}
wvSetPosition -win $_nWave1 {("G1" 6)}
wvSetPosition -win $_nWave1 {("G1" 6)}
wvGetSignalClose -win $_nWave1
wvSelectSignal -win $_nWave1 {( "G1" 4 5 6 )} 
wvSetRadix -win $_nWave1 -format UDec
wvSetCursor -win $_nWave1 349492.410356
wvSelectSignal -win $_nWave1 {( "G1" 6 )} 
wvSelectSignal -win $_nWave1 {( "G1" 3 )} 
wvSelectSignal -win $_nWave1 {( "G1" 5 )} 
wvSelectSignal -win $_nWave1 {( "G1" 4 )} 
wvZoomIn -win $_nWave1
wvZoomIn -win $_nWave1
wvSetCursor -win $_nWave1 349544.222527
wvSetCursor -win $_nWave1 349499.603595
wvZoomAll -win $_nWave1
wvZoomOut -win $_nWave1
wvSetCursor -win $_nWave1 2411355.690469
wvSetCursor -win $_nWave1 2550258.668030
wvSetCursor -win $_nWave1 4383777.971843
wvSetCursor -win $_nWave1 1905748.852145
wvZoomIn -win $_nWave1
wvZoomIn -win $_nWave1
wvSetCursor -win $_nWave1 3416875.720314
wvSetOptions -win $_nWave1 -snap on
wvGetSignalOpen -win $_nWave1
wvGetSignalSetScope -win $_nWave1 "/test_top"
wvGetSignalSetScope -win $_nWave1 "/test_top/top"
wvGetSignalSetScope -win $_nWave1 "/test_top/top/sum_RD"
wvGetSignalSetScope -win $_nWave1 "/test_top/top/sum_RD"
wvSetPosition -win $_nWave1 {("G1" 6)}
wvSetPosition -win $_nWave1 {("G1" 7)}
wvSetPosition -win $_nWave1 {("G1" 7)}
wvAddSignal -win $_nWave1 -clear
wvAddSignal -win $_nWave1 -group {"G1" \
{/test_top/top/sum_RD/sum1\[42:0\]} -color ID_RED5 \
{/test_top/top/sum_RD/sum2\[42:0\]} -color ID_RED5 \
{/test_top/top/sum_RD/sum_all\[43:0\]} -color ID_RED5 \
{/test_top/top/sum_RD/pixel_out\[15:0\]} \
{/test_top/top/sum_RD/temp_pixel_out\[15:0\]} \
{/test_top/top/sum_RD/temp_sum1\[42:0\]} \
{/test_top/top/sum_RD/state\[3:0\]} \
}
wvAddSignal -win $_nWave1 -group {"G2" \
}
wvSelectSignal -win $_nWave1 {( "G1" 7 )} 
wvSetPosition -win $_nWave1 {("G1" 7)}
wvSetPosition -win $_nWave1 {("G1" 6)}
wvSetPosition -win $_nWave1 {("G1" 7)}
wvSetPosition -win $_nWave1 {("G1" 7)}
wvSetPosition -win $_nWave1 {("G1" 7)}
wvSetPosition -win $_nWave1 {("G1" 7)}
wvSetPosition -win $_nWave1 {("G1" 7)}
wvAddSignal -win $_nWave1 -clear
wvAddSignal -win $_nWave1 -group {"G1" \
{/test_top/top/sum_RD/sum1\[42:0\]} -color ID_RED5 \
{/test_top/top/sum_RD/sum2\[42:0\]} -color ID_RED5 \
{/test_top/top/sum_RD/sum_all\[43:0\]} -color ID_RED5 \
{/test_top/top/sum_RD/pixel_out\[15:0\]} \
{/test_top/top/sum_RD/temp_pixel_out\[15:0\]} \
{/test_top/top/sum_RD/temp_sum1\[42:0\]} \
{/test_top/top/sum_RD/state\[3:0\]} \
}
wvAddSignal -win $_nWave1 -group {"G2" \
}
wvSelectSignal -win $_nWave1 {( "G1" 7 )} 
wvSetPosition -win $_nWave1 {("G1" 7)}
wvSetPosition -win $_nWave1 {("G1" 7)}
wvSetPosition -win $_nWave1 {("G1" 7)}
wvGetSignalClose -win $_nWave1
wvZoom -win $_nWave1 3426502.659353 3544776.481831
wvZoom -win $_nWave1 3468119.262496 3479298.440315
wvZoom -win $_nWave1 3469904.024006 3470241.441605
wvSelectSignal -win $_nWave1 {( "G1" 3 )} 
wvSelectSignal -win $_nWave1 {( "G1" 6 )} 
wvSelectSignal -win $_nWave1 {( "G1" 4 )} 
wvZoom -win $_nWave1 3470120.571598 3470148.176056
wvZoomOut -win $_nWave1
wvSetCursor -win $_nWave1 3470131.019195 -snap {("G1" 5)}
wvSetCursor -win $_nWave1 3470129.572098 -snap {("G1" 5)}
wvSetCursor -win $_nWave1 3470142.683668 -snap {("G2" 0)}
wvSetOptions -win $_nWave1 -snap off
wvGetSignalOpen -win $_nWave1
wvGetSignalSetScope -win $_nWave1 "/test_top"
wvGetSignalSetScope -win $_nWave1 "/test_top/top"
wvGetSignalSetScope -win $_nWave1 "/test_top/top/sum_RD"
wvGetSignalSetScope -win $_nWave1 "/test_top/top/sum_RD"
wvSetPosition -win $_nWave1 {("G1" 7)}
wvSetPosition -win $_nWave1 {("G1" 8)}
wvSetPosition -win $_nWave1 {("G1" 8)}
wvAddSignal -win $_nWave1 -clear
wvAddSignal -win $_nWave1 -group {"G1" \
{/test_top/top/sum_RD/sum1\[42:0\]} -color ID_RED5 \
{/test_top/top/sum_RD/sum2\[42:0\]} -color ID_RED5 \
{/test_top/top/sum_RD/sum_all\[43:0\]} -color ID_RED5 \
{/test_top/top/sum_RD/pixel_out\[15:0\]} \
{/test_top/top/sum_RD/temp_pixel_out\[15:0\]} \
{/test_top/top/sum_RD/temp_sum1\[42:0\]} \
{/test_top/top/sum_RD/state\[3:0\]} \
{/test_top/top/sum_RD/sum_saturate\[15:0\]} \
}
wvAddSignal -win $_nWave1 -group {"G2" \
}
wvSelectSignal -win $_nWave1 {( "G1" 8 )} 
wvSetPosition -win $_nWave1 {("G1" 8)}
wvSetPosition -win $_nWave1 {("G1" 7)}
wvSetPosition -win $_nWave1 {("G1" 8)}
wvSetPosition -win $_nWave1 {("G1" 8)}
wvSetPosition -win $_nWave1 {("G1" 8)}
wvGetSignalClose -win $_nWave1
wvSelectSignal -win $_nWave1 {( "G1" 8 )} 
wvSetRadix -win $_nWave1 -format UDec
wvSetCursor -win $_nWave1 3470120.407155
wvSetCursor -win $_nWave1 3470120.231750
wvSetRadix -win $_nWave1 -2Com
wvSetCursor -win $_nWave1 3470128.125002
wvSetOptions -win $_nWave1 -snap on
wvGetSignalOpen -win $_nWave1
wvGetSignalSetScope -win $_nWave1 "/test_top"
wvGetSignalSetScope -win $_nWave1 "/test_top/top"
wvGetSignalSetScope -win $_nWave1 "/test_top/top/sum_RD"
wvGetSignalSetScope -win $_nWave1 "/test_top/top/sum_RD"
wvSetPosition -win $_nWave1 {("G1" 8)}
wvSetPosition -win $_nWave1 {("G1" 9)}
wvSetPosition -win $_nWave1 {("G1" 9)}
wvAddSignal -win $_nWave1 -clear
wvAddSignal -win $_nWave1 -group {"G1" \
{/test_top/top/sum_RD/sum1\[42:0\]} -color ID_RED5 \
{/test_top/top/sum_RD/sum2\[42:0\]} -color ID_RED5 \
{/test_top/top/sum_RD/sum_all\[43:0\]} -color ID_RED5 \
{/test_top/top/sum_RD/pixel_out\[15:0\]} \
{/test_top/top/sum_RD/temp_pixel_out\[15:0\]} \
{/test_top/top/sum_RD/temp_sum1\[42:0\]} \
{/test_top/top/sum_RD/state\[3:0\]} \
{/test_top/top/sum_RD/sum_saturate\[15:0\]} \
{/test_top/top/sum_RD/sum_round\[43:0\]} \
}
wvAddSignal -win $_nWave1 -group {"G2" \
}
wvSelectSignal -win $_nWave1 {( "G1" 9 )} 
wvSetPosition -win $_nWave1 {("G1" 9)}
wvSetPosition -win $_nWave1 {("G1" 8)}
wvSetPosition -win $_nWave1 {("G1" 9)}
wvSetPosition -win $_nWave1 {("G1" 9)}
wvSetPosition -win $_nWave1 {("G1" 9)}
wvGetSignalClose -win $_nWave1
wvSelectSignal -win $_nWave1 {( "G1" 9 )} 
wvSetRadix -win $_nWave1 -format UDec
wvSetRadix -win $_nWave1 -2Com
wvSelectSignal -win $_nWave1 {( "G1" 6 )} 
wvSelectSignal -win $_nWave1 {( "G1" 9 )} 
wvResizeWindow -win $_nWave1 0 23 1536 801
wvSelectSignal -win $_nWave1 {( "G1" 6 )} 
wvCut -win $_nWave1
wvSetPosition -win $_nWave1 {("G2" 0)}
wvSetPosition -win $_nWave1 {("G1" 8)}
wvSetPosition -win $_nWave1 {("G1" 8)}
wvSetPosition -win $_nWave1 {("G1" 8)}
wvSetPosition -win $_nWave1 {("G1" 8)}
wvSelectSignal -win $_nWave1 {( "G1" 1 )} 
wvSelectSignal -win $_nWave1 {( "G1" 2 )} 
wvSetOptions -win $_nWave1 -snap off
wvGetSignalOpen -win $_nWave1
wvGetSignalSetScope -win $_nWave1 "/test_top"
wvGetSignalSetScope -win $_nWave1 "/test_top/top"
wvGetSignalSetScope -win $_nWave1 "/test_top/top/sum_RD"
wvGetSignalSetScope -win $_nWave1 "/test_top/top/sum_RD"
wvSetPosition -win $_nWave1 {("G1" 8)}
wvSetPosition -win $_nWave1 {("G1" 9)}
wvSetPosition -win $_nWave1 {("G1" 9)}
wvAddSignal -win $_nWave1 -clear
wvAddSignal -win $_nWave1 -group {"G1" \
{/test_top/top/sum_RD/sum1\[42:0\]} -color ID_RED5 \
{/test_top/top/sum_RD/sum2\[42:0\]} -color ID_RED5 \
{/test_top/top/sum_RD/sum_all\[43:0\]} -color ID_RED5 \
{/test_top/top/sum_RD/pixel_out\[15:0\]} \
{/test_top/top/sum_RD/temp_pixel_out\[15:0\]} \
{/test_top/top/sum_RD/state\[3:0\]} \
{/test_top/top/sum_RD/sum_saturate\[15:0\]} \
{/test_top/top/sum_RD/sum_round\[43:0\]} \
{/test_top/top/sum_RD/sram_rdata_bias_delay4\[7:0\]} \
}
wvAddSignal -win $_nWave1 -group {"G2" \
}
wvSelectSignal -win $_nWave1 {( "G1" 9 )} 
wvSetPosition -win $_nWave1 {("G1" 9)}
wvSetPosition -win $_nWave1 {("G1" 8)}
wvSetPosition -win $_nWave1 {("G1" 9)}
wvSetPosition -win $_nWave1 {("G1" 9)}
wvSetPosition -win $_nWave1 {("G1" 9)}
wvGetSignalClose -win $_nWave1
wvZoomOut -win $_nWave1
wvZoomOut -win $_nWave1
wvZoomOut -win $_nWave1
wvZoomOut -win $_nWave1
wvZoomOut -win $_nWave1
wvZoomOut -win $_nWave1
wvZoomOut -win $_nWave1
wvZoomOut -win $_nWave1
wvZoomIn -win $_nWave1
wvZoomIn -win $_nWave1
wvZoomIn -win $_nWave1
wvZoomIn -win $_nWave1
wvZoomIn -win $_nWave1
wvZoomIn -win $_nWave1
wvZoomIn -win $_nWave1
wvZoomOut -win $_nWave1
wvZoomOut -win $_nWave1
wvZoomOut -win $_nWave1
wvZoomIn -win $_nWave1
wvZoomIn -win $_nWave1
wvResizeWindow -win $_nWave1 0 23 1536 801
wvZoomIn -win $_nWave1
wvSelectSignal -win $_nWave1 {( "G1" 3 )} 
wvSetCursor -win $_nWave1 3470118.876794
wvSetCursor -win $_nWave1 3470111.675480
wvSetCursor -win $_nWave1 3470121.599242
wvSetCursor -win $_nWave1 3470114.222286
wvSetCursor -win $_nWave1 3470112.290226
wvSetCursor -win $_nWave1 3470130.253991
wvSetCursor -win $_nWave1 3470111.196856
wvSetCursor -win $_nWave1 3470121.384080
wvSetCursor -win $_nWave1 3470150.013693
wvSelectSignal -win $_nWave1 {( "G1" 6 )} 
wvSelectSignal -win $_nWave1 {( "G1" 5 )} 
wvSetCursor -win $_nWave1 3470184.382401
wvZoomAll -win $_nWave1
wvSetCursor -win $_nWave1 5309426.436983
wvZoom -win $_nWave1 3525262.280871 4343516.876605
wvZoom -win $_nWave1 3683587.708365 3838314.830689
wvZoom -win $_nWave1 3728359.232766 3748227.447242
wvZoom -win $_nWave1 3735191.661842 3738127.334693
wvZoom -win $_nWave1 3736389.684887 3736756.321250
wvZoom -win $_nWave1 3736546.400165 3736601.863186
wvZoomOut -win $_nWave1
wvSetCursor -win $_nWave1 3736601.497334
wvSelectSignal -win $_nWave1 {( "G1" 6 )} 
wvZoomOut -win $_nWave1
wvZoomOut -win $_nWave1
wvZoomOut -win $_nWave1
wvZoomOut -win $_nWave1
wvZoomOut -win $_nWave1
wvZoomOut -win $_nWave1
wvZoomOut -win $_nWave1
wvZoomOut -win $_nWave1
wvZoomOut -win $_nWave1
wvZoomOut -win $_nWave1
wvZoomOut -win $_nWave1
wvZoomOut -win $_nWave1
wvZoomOut -win $_nWave1
wvZoomOut -win $_nWave1
wvZoomIn -win $_nWave1
wvZoomIn -win $_nWave1
wvZoomIn -win $_nWave1
wvZoomIn -win $_nWave1
wvZoomIn -win $_nWave1
wvZoomIn -win $_nWave1
wvZoomIn -win $_nWave1
wvZoomIn -win $_nWave1
wvZoomOut -win $_nWave1
wvZoomOut -win $_nWave1
wvZoomOut -win $_nWave1
wvZoomOut -win $_nWave1
wvZoomOut -win $_nWave1
wvZoomOut -win $_nWave1
wvZoomOut -win $_nWave1
wvZoomOut -win $_nWave1
wvZoomOut -win $_nWave1
wvZoomOut -win $_nWave1
wvZoomOut -win $_nWave1
wvZoomOut -win $_nWave1
wvZoomIn -win $_nWave1
wvZoomIn -win $_nWave1
wvZoomOut -win $_nWave1
wvZoomOut -win $_nWave1
wvZoom -win $_nWave1 3514724.199648 3734013.751099
wvSetCursor -win $_nWave1 3620415.205976
wvZoom -win $_nWave1 3613472.001181 3631987.213968
wvZoom -win $_nWave1 3618878.378177 3622281.790202
wvZoom -win $_nWave1 3620189.455104 3620413.954578
wvSelectSignal -win $_nWave1 {( "G1" 1 )} 
wvSetCursor -win $_nWave1 3619644.525576
wvSelectSignal -win $_nWave1 {( "G1" 8 )} 
wvSelectSignal -win $_nWave1 {( "G1" 4 )} 
wvSelectSignal -win $_nWave1 {( "G1" 3 )} 
wvSelectSignal -win $_nWave1 {( "G1" 5 )} 
wvSelectSignal -win $_nWave1 {( "G1" 6 )} 
wvSelectSignal -win $_nWave1 {( "G1" 5 )} 
wvSelectSignal -win $_nWave1 {( "G1" 4 )} 
wvSelectSignal -win $_nWave1 {( "G1" 7 )} 
wvSelectSignal -win $_nWave1 {( "G1" 8 )} 
wvSetCursor -win $_nWave1 3617918.861106
wvSelectSignal -win $_nWave1 {( "G1" 3 )} 
wvSetCursor -win $_nWave1 3617920.835596
wvSetCursor -win $_nWave1 3617912.147841
wvSetCursor -win $_nWave1 3617922.217738
wvSetCursor -win $_nWave1 3617929.523350
wvSetCursor -win $_nWave1 3617917.281514
wvSetCursor -win $_nWave1 3617930.905493
wvSetCursor -win $_nWave1 3617923.599881
wvSetCursor -win $_nWave1 3617927.943758
wvSetCursor -win $_nWave1 3617920.440698
wvSetCursor -win $_nWave1 3617928.536105
wvSelectSignal -win $_nWave1 {( "G1" 1 )} 
wvSetRadix -win $_nWave1 -2Com
wvSetCursor -win $_nWave1 3617921.427942
wvSetCursor -win $_nWave1 3617930.510595
wvSetCursor -win $_nWave1 3617920.835596
wvSetCursor -win $_nWave1 3617912.345290
wvSetCursor -win $_nWave1 3617921.033045
wvSelectSignal -win $_nWave1 {( "G1" 2 )} 
wvSetRadix -win $_nWave1 -2Com
wvSelectSignal -win $_nWave1 {( "G1" 3 )} 
wvSetRadix -win $_nWave1 -2Com
wvSetCursor -win $_nWave1 3617919.256004
wvSelectSignal -win $_nWave1 {( "G1" 8 )} 
wvSetRadix -win $_nWave1 -2Com
wvZoomAll -win $_nWave1
wvZoom -win $_nWave1 3580632.892612 3636003.504354
wvZoom -win $_nWave1 3617400.537172 3618471.912245
wvZoom -win $_nWave1 3617899.946837 3617959.310627
wvSetCursor -win $_nWave1 3617920.309086
wvSelectSignal -win $_nWave1 {( "G1" 3 )} 
wvSetCursor -win $_nWave1 3617905.324558
wvSetCursor -win $_nWave1 3617903.183912
wvSetCursor -win $_nWave1 3617908.927110
wvSetCursor -win $_nWave1 3617920.309086
wvSetCursor -win $_nWave1 3617928.036299
wvSetCursor -win $_nWave1 3617915.296840
wvSetCursor -win $_nWave1 3617941.245656
wvSelectSignal -win $_nWave1 {( "G1" 4 )} 
wvSelectSignal -win $_nWave1 {( "G1" 3 )} 
wvSetCursor -win $_nWave1 3617921.614358
wvSelectSignal -win $_nWave1 {( "G1" 6 )} 
wvSelectSignal -win $_nWave1 {( "G1" 5 )} 
wvSelectSignal -win $_nWave1 {( "G1" 8 )} 
wvSelectSignal -win $_nWave1 {( "G1" 3 )} 
wvSelectSignal -win $_nWave1 {( "G1" 4 )} 
wvSetCursor -win $_nWave1 3617920.152453
wvSetCursor -win $_nWave1 3617920.152453
wvSetCursor -win $_nWave1 3617920.152453
wvSetCursor -win $_nWave1 3617920.152453
wvSelectSignal -win $_nWave1 {( "G1" 3 )} 
wvSetRadix -win $_nWave1 -Unsigned
wvSelectSignal -win $_nWave1 {( "G1" 2 )} 
wvSelectSignal -win $_nWave1 {( "G1" 3 )} 
wvSelectSignal -win $_nWave1 {( "G1" 3 )} 
wvSetRadix -win $_nWave1 -format Bin
wvSelectSignal -win $_nWave1 {( "G1" 8 )} 
wvSelectSignal -win $_nWave1 {( "G1" 8 )} 
wvSetRadix -win $_nWave1 -format Bin
wvResizeWindow -win $_nWave1 0 23 1536 801
wvSelectSignal -win $_nWave1 {( "G1" 8 )} 
wvSetRadix -win $_nWave1 -format UDec
wvResizeWindow -win $_nWave1 0 23 1536 801
wvSelectSignal -win $_nWave1 {( "G1" 3 )} 
wvSetRadix -win $_nWave1 -2Com
wvSelectSignal -win $_nWave1 {( "G1" 3 )} 
wvSetRadix -win $_nWave1 -format UDec
wvSelectSignal -win $_nWave1 {( "G1" 7 )} 
wvSetCursor -win $_nWave1 3617910.837321
wvSetCursor -win $_nWave1 3617921.413658
wvSetPosition -win $_nWave1 {("G1" 9)}
wvSetPosition -win $_nWave1 {("G1" 9)}
wvSetPosition -win $_nWave1 {("G1" 9)}
wvDisplayGridCount -win $_nWave1 -off
wvGetSignalClose -win $_nWave1
wvReloadFile -win $_nWave1
wvSelectSignal -win $_nWave1 {( "G1" 8 )} 
wvSetCursor -win $_nWave1 3617903.507188
wvExit
