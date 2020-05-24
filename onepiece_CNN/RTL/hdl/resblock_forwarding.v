module resblock_forwarding #(
    parameter CH_NUM = 24,
    parameter ACT_PER_ADDR = 4,
    parameter BW_PER_ACT = 16,
    parameter WEIGHT_PER_ADDR = 216, 
    parameter BIAS_PER_ADDR = 1,
    parameter BW_PER_WEIGHT = 8,
    parameter BW_PER_BIAS = 8
)
(
    input [1:0] map_type_delay4,
    input [6:0] fmap_idx_delay4,
    input [CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1:0] sram_rdata_b0_delay,  //[1919:0]
    input [CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1:0] sram_rdata_b1_delay,
    input [CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1:0] sram_rdata_b2_delay,
    input [CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1:0] sram_rdata_b3_delay,
    output reg signed [BW_PER_ACT-1:0] LU_forwarding,
    output reg signed [BW_PER_ACT-1:0] RU_forwarding,
    output reg signed [BW_PER_ACT-1:0] LD_forwarding,
    output reg signed [BW_PER_ACT-1:0] RD_forwarding
);

//RESBLOCK forwarding
always@*  begin
    case(map_type_delay4)
        2'd0:  begin
            LU_forwarding = sram_rdata_b0_delay[(CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1-3*BW_PER_ACT-fmap_idx_delay4*4*BW_PER_ACT)-:BW_PER_ACT];
            RU_forwarding = sram_rdata_b1_delay[(CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1-2*BW_PER_ACT-fmap_idx_delay4*4*BW_PER_ACT)-:BW_PER_ACT];
            LD_forwarding = sram_rdata_b2_delay[(CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1-BW_PER_ACT-fmap_idx_delay4*4*BW_PER_ACT)-:BW_PER_ACT];
            RD_forwarding = sram_rdata_b3_delay[(CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1-fmap_idx_delay4*4*BW_PER_ACT)-:BW_PER_ACT];
        end
        2'd1:  begin
            LU_forwarding = sram_rdata_b1_delay[(CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1-3*BW_PER_ACT-fmap_idx_delay4*4*BW_PER_ACT)-:BW_PER_ACT];
            RU_forwarding = sram_rdata_b0_delay[(CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1-2*BW_PER_ACT-fmap_idx_delay4*4*BW_PER_ACT)-:BW_PER_ACT];
            LD_forwarding = sram_rdata_b3_delay[(CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1-BW_PER_ACT-fmap_idx_delay4*4*BW_PER_ACT)-:BW_PER_ACT];
            RD_forwarding = sram_rdata_b2_delay[(CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1-fmap_idx_delay4*4*BW_PER_ACT)-:BW_PER_ACT];
        end
        2'd2:  begin
            LU_forwarding = sram_rdata_b2_delay[(CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1-3*BW_PER_ACT-fmap_idx_delay4*4*BW_PER_ACT)-:BW_PER_ACT];
            RU_forwarding = sram_rdata_b3_delay[(CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1-2*BW_PER_ACT-fmap_idx_delay4*4*BW_PER_ACT)-:BW_PER_ACT];
            LD_forwarding = sram_rdata_b0_delay[(CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1-BW_PER_ACT-fmap_idx_delay4*4*BW_PER_ACT)-:BW_PER_ACT];
            RD_forwarding = sram_rdata_b1_delay[(CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1-fmap_idx_delay4*4*BW_PER_ACT)-:BW_PER_ACT];
        end
        default:  begin
            LU_forwarding = sram_rdata_b3_delay[(CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1-3*BW_PER_ACT-fmap_idx_delay4*4*BW_PER_ACT)-:BW_PER_ACT];
            RU_forwarding = sram_rdata_b2_delay[(CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1-2*BW_PER_ACT-fmap_idx_delay4*4*BW_PER_ACT)-:BW_PER_ACT];
            LD_forwarding = sram_rdata_b1_delay[(CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1-BW_PER_ACT-fmap_idx_delay4*4*BW_PER_ACT)-:BW_PER_ACT];
            RD_forwarding = sram_rdata_b0_delay[(CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1-fmap_idx_delay4*4*BW_PER_ACT)-:BW_PER_ACT];
        end
    endcase
end

endmodule