module write_LD_case #(
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
        7'd0:  sram_bytemask   = { {2{1'b1}} , {1'b0}, {93{1'b1}} };
        7'd1:  sram_bytemask   = { {6{1'b1}} , {1'b0}, {89{1'b1}} };
        7'd2:  sram_bytemask   = { {10{1'b1}}, {1'b0}, {85{1'b1}} };
        7'd3:  sram_bytemask   = { {14{1'b1}}, {1'b0}, {81{1'b1}} };
        7'd4:  sram_bytemask   = { {18{1'b1}}, {1'b0}, {77{1'b1}} };
        7'd5:  sram_bytemask   = { {22{1'b1}}, {1'b0}, {73{1'b1}} };
        7'd6:  sram_bytemask   = { {26{1'b1}}, {1'b0}, {69{1'b1}} };
        7'd7:  sram_bytemask   = { {30{1'b1}}, {1'b0}, {65{1'b1}} };
        7'd8:  sram_bytemask   = { {34{1'b1}}, {1'b0}, {61{1'b1}} };
        7'd9:  sram_bytemask   = { {38{1'b1}}, {1'b0}, {57{1'b1}} };
        7'd10: sram_bytemask   = { {42{1'b1}}, {1'b0}, {53{1'b1}} };
        7'd11: sram_bytemask   = { {46{1'b1}}, {1'b0}, {49{1'b1}} };
        7'd12: sram_bytemask   = { {50{1'b1}}, {1'b0}, {45{1'b1}} };
        7'd13: sram_bytemask   = { {54{1'b1}}, {1'b0}, {41{1'b1}} };
        7'd14: sram_bytemask   = { {58{1'b1}}, {1'b0}, {37{1'b1}} };
        7'd15: sram_bytemask   = { {62{1'b1}}, {1'b0}, {33{1'b1}} };
        7'd16: sram_bytemask   = { {66{1'b1}}, {1'b0}, {29{1'b1}} };
        7'd17: sram_bytemask   = { {70{1'b1}}, {1'b0}, {25{1'b1}} };
        7'd18: sram_bytemask   = { {74{1'b1}}, {1'b0}, {21{1'b1}} };
        7'd19: sram_bytemask   = { {78{1'b1}}, {1'b0}, {17{1'b1}} };
        7'd20: sram_bytemask   = { {82{1'b1}}, {1'b0}, {13{1'b1}} };
        7'd21: sram_bytemask   = { {86{1'b1}}, {1'b0}, {9{1'b1}}  };
        7'd22: sram_bytemask   = { {90{1'b1}}, {1'b0}, {5{1'b1}}  };
        7'd23: sram_bytemask   = { {94{1'b1}}, {1'b0}, {1{1'b1}}  };
        default: sram_bytemask = { {2{1'b1}} , {1'b0}, {93{1'b1}} };
    endcase
end

endmodule