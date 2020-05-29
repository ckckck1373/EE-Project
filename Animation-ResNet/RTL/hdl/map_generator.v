module map_generator #(
    parameter CH_NUM = 24,
    parameter ACT_PER_ADDR = 4,
    parameter BW_PER_ACT = 16,
    parameter WEIGHT_PER_ADDR = 216, 
    parameter BIAS_PER_ADDR = 1,
    parameter BW_PER_WEIGHT = 8,
    parameter BW_PER_BIAS   = 8,
    parameter BASE_BW = 11
)
(
    input [1:0] map_type_delay,
    input [3:0] state,
    input [CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1:0] sram_rdata_a0_delay,  //[1919:0]
    input [CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1:0] sram_rdata_a1_delay,
    input [CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1:0] sram_rdata_a2_delay,
    input [CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1:0] sram_rdata_a3_delay,
    input [CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1:0] sram_rdata_b0_delay,  //[1919:0]
    input [CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1:0] sram_rdata_b1_delay,
    input [CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1:0] sram_rdata_b2_delay,
    input [CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1:0] sram_rdata_b3_delay,
    input [BASE_BW-1:0] base_0,
    input [BASE_BW-1:0] base_1,
    input [BASE_BW-1:0] base_2,
    input [BASE_BW-1:0] base_3,
    output reg signed [BW_PER_ACT-1:0] map_0,
    output reg signed [BW_PER_ACT-1:0] map_1,
    output reg signed [BW_PER_ACT-1:0] map_2,
    output reg signed [BW_PER_ACT-1:0] map_3,
    output reg signed [BW_PER_ACT-1:0] map_4,
    output reg signed [BW_PER_ACT-1:0] map_5,
    output reg signed [BW_PER_ACT-1:0] map_6,
    output reg signed [BW_PER_ACT-1:0] map_7,
    output reg signed [BW_PER_ACT-1:0] map_8,
    output reg signed [BW_PER_ACT-1:0] map_9,
    output reg signed [BW_PER_ACT-1:0] map_10,
    output reg signed [BW_PER_ACT-1:0] map_11,
    output reg signed [BW_PER_ACT-1:0] map_12,
    output reg signed [BW_PER_ACT-1:0] map_13,
    output reg signed [BW_PER_ACT-1:0] map_14,
    output reg signed [BW_PER_ACT-1:0] map_15
);
parameter IDLE = 0, PADDING = 1, CONV1 = 2, RES_1 = 3, RES_2 = 4, UP_1 = 5, UP_2 = 6, CONV2 = 7, FINISH = 8;

//generate map depends on different map type
always @*  begin
    if(state==CONV1 || state==RES_2 || state==UP_2)  begin  //read data from SRAM_A
        case(map_type_delay)
            2'd0:  begin
                map_0  = sram_rdata_a0_delay[base_0-:BW_PER_ACT];
                map_1  = sram_rdata_a0_delay[base_1-:BW_PER_ACT];
                map_2  = sram_rdata_a1_delay[base_0-:BW_PER_ACT];
                map_3  = sram_rdata_a1_delay[base_1-:BW_PER_ACT];
                map_4  = sram_rdata_a0_delay[base_2-:BW_PER_ACT];
                map_5  = sram_rdata_a0_delay[base_3-:BW_PER_ACT];
                map_6  = sram_rdata_a1_delay[base_2-:BW_PER_ACT];
                map_7  = sram_rdata_a1_delay[base_3-:BW_PER_ACT]; 
                map_8  = sram_rdata_a2_delay[base_0-:BW_PER_ACT];     
                map_9  = sram_rdata_a2_delay[base_1-:BW_PER_ACT];                         
                map_10 = sram_rdata_a3_delay[base_0-:BW_PER_ACT];                          
                map_11 = sram_rdata_a3_delay[base_1-:BW_PER_ACT];                          
                map_12 = sram_rdata_a2_delay[base_2-:BW_PER_ACT];                        
                map_13 = sram_rdata_a2_delay[base_3-:BW_PER_ACT];                        
                map_14 = sram_rdata_a3_delay[base_2-:BW_PER_ACT];                        
                map_15 = sram_rdata_a3_delay[base_3-:BW_PER_ACT];
            end
            2'd1:  begin
                map_0  = sram_rdata_a1_delay[base_0-:BW_PER_ACT];
                map_1  = sram_rdata_a1_delay[base_1-:BW_PER_ACT];
                map_2  = sram_rdata_a0_delay[base_0-:BW_PER_ACT];
                map_3  = sram_rdata_a0_delay[base_1-:BW_PER_ACT];
                map_4  = sram_rdata_a1_delay[base_2-:BW_PER_ACT];
                map_5  = sram_rdata_a1_delay[base_3-:BW_PER_ACT];
                map_6  = sram_rdata_a0_delay[base_2-:BW_PER_ACT];
                map_7  = sram_rdata_a0_delay[base_3-:BW_PER_ACT]; 
                map_8  = sram_rdata_a3_delay[base_0-:BW_PER_ACT];     
                map_9  = sram_rdata_a3_delay[base_1-:BW_PER_ACT];                         
                map_10 = sram_rdata_a2_delay[base_0-:BW_PER_ACT];                          
                map_11 = sram_rdata_a2_delay[base_1-:BW_PER_ACT];                          
                map_12 = sram_rdata_a3_delay[base_2-:BW_PER_ACT];                        
                map_13 = sram_rdata_a3_delay[base_3-:BW_PER_ACT];                        
                map_14 = sram_rdata_a2_delay[base_2-:BW_PER_ACT];                        
                map_15 = sram_rdata_a2_delay[base_3-:BW_PER_ACT];            
            end
            2'd2:  begin
                map_0  = sram_rdata_a2_delay[base_0-:BW_PER_ACT];
                map_1  = sram_rdata_a2_delay[base_1-:BW_PER_ACT];
                map_2  = sram_rdata_a3_delay[base_0-:BW_PER_ACT];
                map_3  = sram_rdata_a3_delay[base_1-:BW_PER_ACT];
                map_4  = sram_rdata_a2_delay[base_2-:BW_PER_ACT];
                map_5  = sram_rdata_a2_delay[base_3-:BW_PER_ACT];
                map_6  = sram_rdata_a3_delay[base_2-:BW_PER_ACT];
                map_7  = sram_rdata_a3_delay[base_3-:BW_PER_ACT]; 
                map_8  = sram_rdata_a0_delay[base_0-:BW_PER_ACT];     
                map_9  = sram_rdata_a0_delay[base_1-:BW_PER_ACT];                         
                map_10 = sram_rdata_a1_delay[base_0-:BW_PER_ACT];                          
                map_11 = sram_rdata_a1_delay[base_1-:BW_PER_ACT];                          
                map_12 = sram_rdata_a0_delay[base_2-:BW_PER_ACT];                        
                map_13 = sram_rdata_a0_delay[base_3-:BW_PER_ACT];                        
                map_14 = sram_rdata_a1_delay[base_2-:BW_PER_ACT];                        
                map_15 = sram_rdata_a1_delay[base_3-:BW_PER_ACT];
            end
            default:  begin  //2'd3
                map_0  = sram_rdata_a3_delay[base_0-:BW_PER_ACT];
                map_1  = sram_rdata_a3_delay[base_1-:BW_PER_ACT];
                map_2  = sram_rdata_a2_delay[base_0-:BW_PER_ACT];
                map_3  = sram_rdata_a2_delay[base_1-:BW_PER_ACT];
                map_4  = sram_rdata_a3_delay[base_2-:BW_PER_ACT];
                map_5  = sram_rdata_a3_delay[base_3-:BW_PER_ACT];
                map_6  = sram_rdata_a2_delay[base_2-:BW_PER_ACT];
                map_7  = sram_rdata_a2_delay[base_3-:BW_PER_ACT]; 
                map_8  = sram_rdata_a1_delay[base_0-:BW_PER_ACT];     
                map_9  = sram_rdata_a1_delay[base_1-:BW_PER_ACT];                         
                map_10 = sram_rdata_a0_delay[base_0-:BW_PER_ACT];                          
                map_11 = sram_rdata_a0_delay[base_1-:BW_PER_ACT];                          
                map_12 = sram_rdata_a1_delay[base_2-:BW_PER_ACT];                        
                map_13 = sram_rdata_a1_delay[base_3-:BW_PER_ACT];                        
                map_14 = sram_rdata_a0_delay[base_2-:BW_PER_ACT];                        
                map_15 = sram_rdata_a0_delay[base_3-:BW_PER_ACT];
            end
        endcase
    end
    else if(state==RES_1 || state==UP_1 || state==CONV2)  begin  //read data from SRAM_B
        case(map_type_delay)
            2'd0:  begin
                map_0  = sram_rdata_b0_delay[base_0-:BW_PER_ACT];
                map_1  = sram_rdata_b0_delay[base_1-:BW_PER_ACT];
                map_2  = sram_rdata_b1_delay[base_0-:BW_PER_ACT];
                map_3  = sram_rdata_b1_delay[base_1-:BW_PER_ACT];
                map_4  = sram_rdata_b0_delay[base_2-:BW_PER_ACT];
                map_5  = sram_rdata_b0_delay[base_3-:BW_PER_ACT];
                map_6  = sram_rdata_b1_delay[base_2-:BW_PER_ACT];
                map_7  = sram_rdata_b1_delay[base_3-:BW_PER_ACT]; 
                map_8  = sram_rdata_b2_delay[base_0-:BW_PER_ACT];     
                map_9  = sram_rdata_b2_delay[base_1-:BW_PER_ACT];                         
                map_10 = sram_rdata_b3_delay[base_0-:BW_PER_ACT];                          
                map_11 = sram_rdata_b3_delay[base_1-:BW_PER_ACT];                          
                map_12 = sram_rdata_b2_delay[base_2-:BW_PER_ACT];                        
                map_13 = sram_rdata_b2_delay[base_3-:BW_PER_ACT];                        
                map_14 = sram_rdata_b3_delay[base_2-:BW_PER_ACT];                        
                map_15 = sram_rdata_b3_delay[base_3-:BW_PER_ACT];
            end
            2'd1:  begin
                map_0  = sram_rdata_b1_delay[base_0-:BW_PER_ACT];
                map_1  = sram_rdata_b1_delay[base_1-:BW_PER_ACT];
                map_2  = sram_rdata_b0_delay[base_0-:BW_PER_ACT];
                map_3  = sram_rdata_b0_delay[base_1-:BW_PER_ACT];
                map_4  = sram_rdata_b1_delay[base_2-:BW_PER_ACT];
                map_5  = sram_rdata_b1_delay[base_3-:BW_PER_ACT];
                map_6  = sram_rdata_b0_delay[base_2-:BW_PER_ACT];
                map_7  = sram_rdata_b0_delay[base_3-:BW_PER_ACT]; 
                map_8  = sram_rdata_b3_delay[base_0-:BW_PER_ACT];     
                map_9  = sram_rdata_b3_delay[base_1-:BW_PER_ACT];                         
                map_10 = sram_rdata_b2_delay[base_0-:BW_PER_ACT];                          
                map_11 = sram_rdata_b2_delay[base_1-:BW_PER_ACT];                          
                map_12 = sram_rdata_b3_delay[base_2-:BW_PER_ACT];                        
                map_13 = sram_rdata_b3_delay[base_3-:BW_PER_ACT];                        
                map_14 = sram_rdata_b2_delay[base_2-:BW_PER_ACT];                        
                map_15 = sram_rdata_b2_delay[base_3-:BW_PER_ACT];            
            end
            2'd2:  begin
                map_0  = sram_rdata_b2_delay[base_0-:BW_PER_ACT];
                map_1  = sram_rdata_b2_delay[base_1-:BW_PER_ACT];
                map_2  = sram_rdata_b3_delay[base_0-:BW_PER_ACT];
                map_3  = sram_rdata_b3_delay[base_1-:BW_PER_ACT];
                map_4  = sram_rdata_b2_delay[base_2-:BW_PER_ACT];
                map_5  = sram_rdata_b2_delay[base_3-:BW_PER_ACT];
                map_6  = sram_rdata_b3_delay[base_2-:BW_PER_ACT];
                map_7  = sram_rdata_b3_delay[base_3-:BW_PER_ACT]; 
                map_8  = sram_rdata_b0_delay[base_0-:BW_PER_ACT];     
                map_9  = sram_rdata_b0_delay[base_1-:BW_PER_ACT];                         
                map_10 = sram_rdata_b1_delay[base_0-:BW_PER_ACT];                          
                map_11 = sram_rdata_b1_delay[base_1-:BW_PER_ACT];                          
                map_12 = sram_rdata_b0_delay[base_2-:BW_PER_ACT];                        
                map_13 = sram_rdata_b0_delay[base_3-:BW_PER_ACT];                        
                map_14 = sram_rdata_b1_delay[base_2-:BW_PER_ACT];                        
                map_15 = sram_rdata_b1_delay[base_3-:BW_PER_ACT];
            end
            default:  begin  //2'd3
                map_0  = sram_rdata_b3_delay[base_0-:BW_PER_ACT];
                map_1  = sram_rdata_b3_delay[base_1-:BW_PER_ACT];
                map_2  = sram_rdata_b2_delay[base_0-:BW_PER_ACT];
                map_3  = sram_rdata_b2_delay[base_1-:BW_PER_ACT];
                map_4  = sram_rdata_b3_delay[base_2-:BW_PER_ACT];
                map_5  = sram_rdata_b3_delay[base_3-:BW_PER_ACT];
                map_6  = sram_rdata_b2_delay[base_2-:BW_PER_ACT];
                map_7  = sram_rdata_b2_delay[base_3-:BW_PER_ACT]; 
                map_8  = sram_rdata_b1_delay[base_0-:BW_PER_ACT];     
                map_9  = sram_rdata_b1_delay[base_1-:BW_PER_ACT];                         
                map_10 = sram_rdata_b0_delay[base_0-:BW_PER_ACT];                          
                map_11 = sram_rdata_b0_delay[base_1-:BW_PER_ACT];                          
                map_12 = sram_rdata_b1_delay[base_2-:BW_PER_ACT];                        
                map_13 = sram_rdata_b1_delay[base_3-:BW_PER_ACT];                        
                map_14 = sram_rdata_b0_delay[base_2-:BW_PER_ACT];                        
                map_15 = sram_rdata_b0_delay[base_3-:BW_PER_ACT];
            end
        endcase
    end
    else  begin
        map_0  = 0;
        map_1  = 0;
        map_2  = 0;
        map_3  = 0;
        map_4  = 0;
        map_5  = 0;
        map_6  = 0;
        map_7  = 0;
        map_8  = 0;
        map_9  = 0;
        map_10 = 0;
        map_11 = 0;
        map_12 = 0;
        map_13 = 0;
        map_14 = 0;
        map_15 = 0;
    end
end

endmodule