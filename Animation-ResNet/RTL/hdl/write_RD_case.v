module write_RD_case #(
    parameter CH_NUM = 24,
    parameter ACT_PER_ADDR = 4,
    parameter BW_PER_ACT = 16,
    parameter WEIGHT_PER_ADDR = 216, 
    parameter BIAS_PER_ADDR = 1,
    parameter BW_PER_WEIGHT = 8,
    parameter BW_PER_BIAS = 8
)
(
    input [6:0] fmap_idx_delay5,
    output reg [CH_NUM*ACT_PER_ADDR-1:0] sram_bytemask
);

always@*  begin
    case(fmap_idx_delay5)
        7'd0:  sram_bytemask   = { {3{1'b1}} , {1'b0}, {92{1'b1}} };
        7'd1:  sram_bytemask   = { {7{1'b1}} , {1'b0}, {88{1'b1}} };
        7'd2:  sram_bytemask   = { {11{1'b1}}, {1'b0}, {84{1'b1}} };
        7'd3:  sram_bytemask   = { {15{1'b1}}, {1'b0}, {80{1'b1}} };
        7'd4:  sram_bytemask   = { {19{1'b1}}, {1'b0}, {76{1'b1}} };
        7'd5:  sram_bytemask   = { {23{1'b1}}, {1'b0}, {72{1'b1}} };
        7'd6:  sram_bytemask   = { {27{1'b1}}, {1'b0}, {68{1'b1}} };
        7'd7:  sram_bytemask   = { {31{1'b1}}, {1'b0}, {64{1'b1}} };
        7'd8:  sram_bytemask   = { {35{1'b1}}, {1'b0}, {60{1'b1}} };
        7'd9:  sram_bytemask   = { {39{1'b1}}, {1'b0}, {56{1'b1}} };
        7'd10: sram_bytemask   = { {43{1'b1}}, {1'b0}, {52{1'b1}} };
        7'd11: sram_bytemask   = { {47{1'b1}}, {1'b0}, {48{1'b1}} };
        7'd12: sram_bytemask   = { {51{1'b1}}, {1'b0}, {44{1'b1}} };
        7'd13: sram_bytemask   = { {55{1'b1}}, {1'b0}, {40{1'b1}} };
        7'd14: sram_bytemask   = { {59{1'b1}}, {1'b0}, {36{1'b1}} };
        7'd15: sram_bytemask   = { {63{1'b1}}, {1'b0}, {32{1'b1}} };
        7'd16: sram_bytemask   = { {67{1'b1}}, {1'b0}, {28{1'b1}} };
        7'd17: sram_bytemask   = { {71{1'b1}}, {1'b0}, {24{1'b1}} };
        7'd18: sram_bytemask   = { {75{1'b1}}, {1'b0}, {20{1'b1}} };
        7'd19: sram_bytemask   = { {79{1'b1}}, {1'b0}, {16{1'b1}} };
        7'd20: sram_bytemask   = { {83{1'b1}}, {1'b0}, {12{1'b1}} };
        7'd21: sram_bytemask   = { {87{1'b1}}, {1'b0}, {8{1'b1}}  };
        7'd22: sram_bytemask   = { {91{1'b1}}, {1'b0}, {4{1'b1}}  };
        7'd23: sram_bytemask   = { {95{1'b1}}, {1'b0}             };
        default: sram_bytemask = { {3{1'b1}} , {1'b0}, {92{1'b1}} };
    endcase
end

endmodule