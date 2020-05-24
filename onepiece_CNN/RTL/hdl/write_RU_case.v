module write_RU_case #(
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
        7'd0:  sram_bytemask   = { {1{1'b1}} , {1'b0}, {94{1'b1}} };
        7'd1:  sram_bytemask   = { {5{1'b1}} , {1'b0}, {90{1'b1}} };
        7'd2:  sram_bytemask   = { {9{1'b1}} , {1'b0}, {86{1'b1}} };
        7'd3:  sram_bytemask   = { {13{1'b1}}, {1'b0}, {82{1'b1}} };
        7'd4:  sram_bytemask   = { {17{1'b1}}, {1'b0}, {78{1'b1}} };
        7'd5:  sram_bytemask   = { {21{1'b1}}, {1'b0}, {74{1'b1}} };
        7'd6:  sram_bytemask   = { {25{1'b1}}, {1'b0}, {70{1'b1}} };
        7'd7:  sram_bytemask   = { {29{1'b1}}, {1'b0}, {66{1'b1}} };
        7'd8:  sram_bytemask   = { {33{1'b1}}, {1'b0}, {62{1'b1}} };
        7'd9:  sram_bytemask   = { {37{1'b1}}, {1'b0}, {58{1'b1}} };
        7'd10: sram_bytemask   = { {41{1'b1}}, {1'b0}, {54{1'b1}} };
        7'd11: sram_bytemask   = { {45{1'b1}}, {1'b0}, {50{1'b1}} };
        7'd12: sram_bytemask   = { {49{1'b1}}, {1'b0}, {46{1'b1}} };
        7'd13: sram_bytemask   = { {53{1'b1}}, {1'b0}, {42{1'b1}} };
        7'd14: sram_bytemask   = { {57{1'b1}}, {1'b0}, {38{1'b1}} };
        7'd15: sram_bytemask   = { {61{1'b1}}, {1'b0}, {34{1'b1}} };
        7'd16: sram_bytemask   = { {65{1'b1}}, {1'b0}, {30{1'b1}} };
        7'd17: sram_bytemask   = { {69{1'b1}}, {1'b0}, {26{1'b1}} };
        7'd18: sram_bytemask   = { {73{1'b1}}, {1'b0}, {22{1'b1}} };
        7'd19: sram_bytemask   = { {77{1'b1}}, {1'b0}, {18{1'b1}} };
        7'd20: sram_bytemask   = { {81{1'b1}}, {1'b0}, {14{1'b1}} };
        7'd21: sram_bytemask   = { {85{1'b1}}, {1'b0}, {10{1'b1}} };
        7'd22: sram_bytemask   = { {89{1'b1}}, {1'b0}, {6{1'b1}}  };
        7'd23: sram_bytemask   = { {93{1'b1}}, {1'b0}, {2{1'b1}}  };
        default: sram_bytemask = { {1{1'b1}} , {1'b0}, {94{1'b1}} };
    endcase
end

endmodule