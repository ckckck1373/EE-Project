module write_LU_case #(
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
        7'd0:  sram_bytemask   = { {1'b0}    , {95{1'b1}}       };
        7'd1:  sram_bytemask   = { {4{1'b1}} , {1'b0}, {91{1'b1}} };
        7'd2:  sram_bytemask   = { {8{1'b1}} , {1'b0}, {87{1'b1}} };
        7'd3:  sram_bytemask   = { {12{1'b1}}, {1'b0}, {83{1'b1}} };
        7'd4:  sram_bytemask   = { {16{1'b1}}, {1'b0}, {79{1'b1}} };
        7'd5:  sram_bytemask   = { {20{1'b1}}, {1'b0}, {75{1'b1}} };
        7'd6:  sram_bytemask   = { {24{1'b1}}, {1'b0}, {71{1'b1}} };
        7'd7:  sram_bytemask   = { {28{1'b1}}, {1'b0}, {67{1'b1}} };
        7'd8:  sram_bytemask   = { {32{1'b1}}, {1'b0}, {63{1'b1}} };
        7'd9:  sram_bytemask   = { {36{1'b1}}, {1'b0}, {59{1'b1}} };
        7'd10: sram_bytemask   = { {40{1'b1}}, {1'b0}, {55{1'b1}} };
        7'd11: sram_bytemask   = { {44{1'b1}}, {1'b0}, {51{1'b1}} };
        7'd12: sram_bytemask   = { {48{1'b1}}, {1'b0}, {47{1'b1}} };
        7'd13: sram_bytemask   = { {52{1'b1}}, {1'b0}, {43{1'b1}} };
        7'd14: sram_bytemask   = { {56{1'b1}}, {1'b0}, {39{1'b1}} };
        7'd15: sram_bytemask   = { {60{1'b1}}, {1'b0}, {35{1'b1}} };
        7'd16: sram_bytemask   = { {64{1'b1}}, {1'b0}, {31{1'b1}} };
        7'd17: sram_bytemask   = { {68{1'b1}}, {1'b0}, {27{1'b1}} };
        7'd18: sram_bytemask   = { {72{1'b1}}, {1'b0}, {23{1'b1}} };
        7'd19: sram_bytemask   = { {76{1'b1}}, {1'b0}, {19{1'b1}} };
        7'd20: sram_bytemask   = { {80{1'b1}}, {1'b0}, {15{1'b1}} };
        7'd21: sram_bytemask   = { {84{1'b1}}, {1'b0}, {11{1'b1}} };
        7'd22: sram_bytemask   = { {88{1'b1}}, {1'b0}, {7{1'b1}}  };
        7'd23: sram_bytemask   = { {92{1'b1}}, {1'b0}, {3{1'b1}}  };
        default: sram_bytemask = { {1'b0}    , {95{1'b1}}         };
    endcase
end

endmodule