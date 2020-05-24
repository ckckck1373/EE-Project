module conv_mul #(
    parameter CH_NUM = 24,
    parameter ACT_PER_ADDR = 4,
    parameter BW_PER_ACT = 16,
    parameter WEIGHT_PER_ADDR = 216, 
    parameter BIAS_PER_ADDR = 1,
    parameter BW_PER_WEIGHT = 8,
    parameter BW_PER_BIAS = 8,
    parameter BASE_BW = 11
)
(
    input clk,
    input rst_n,
    input [WEIGHT_PER_ADDR*BW_PER_WEIGHT-1:0] sram_rdata_weight_delay,  //[1727:0]
    input [BASE_BW-1:0] base,
    input signed [BW_PER_ACT-1:0] map_0,  //pixel
    input signed [BW_PER_ACT-1:0] map_1,
    input signed [BW_PER_ACT-1:0] map_2,
    input signed [BW_PER_ACT-1:0] map_3,
    input signed [BW_PER_ACT-1:0] map_4,
    input signed [BW_PER_ACT-1:0] map_5,
    input signed [BW_PER_ACT-1:0] map_6,
    input signed [BW_PER_ACT-1:0] map_7,
    input signed [BW_PER_ACT-1:0] map_8,
    input signed [BW_PER_ACT-1:0] map_9,
    input signed [BW_PER_ACT-1:0] map_10,
    input signed [BW_PER_ACT-1:0] map_11,
    input signed [BW_PER_ACT-1:0] map_12,
    input signed [BW_PER_ACT-1:0] map_13,
    input signed [BW_PER_ACT-1:0] map_14,
    input signed [BW_PER_ACT-1:0] map_15,
    output reg signed [BW_PER_ACT + BW_PER_WEIGHT + 8 - 1:0] LU_sum,
    output reg signed [BW_PER_ACT + BW_PER_WEIGHT + 8 - 1:0] RU_sum,
    output reg signed [BW_PER_ACT + BW_PER_WEIGHT + 8 - 1:0] LD_sum,
    output reg signed [BW_PER_ACT + BW_PER_WEIGHT + 8 - 1:0] RD_sum
);

// weight
reg signed [BW_PER_WEIGHT - 1:0] w0, w1, w2, w3, w4, w5, w6, w7, w8;


// ex: 20-bit * 8-bit (pixel * weight) -> 28-bit
reg signed [BW_PER_ACT + BW_PER_WEIGHT - 1:0] LU_0, LU_1, LU_2, LU_3, LU_4, LU_5, LU_6, LU_7, LU_8, temp_LU_0, temp_LU_1, temp_LU_2, temp_LU_3, temp_LU_4, temp_LU_5, temp_LU_6, temp_LU_7, temp_LU_8;
reg signed [BW_PER_ACT + BW_PER_WEIGHT - 1:0] RU_0, RU_1, RU_2, RU_3, RU_4, RU_5, RU_6, RU_7, RU_8, temp_RU_0, temp_RU_1, temp_RU_2, temp_RU_3, temp_RU_4, temp_RU_5, temp_RU_6, temp_RU_7, temp_RU_8;
reg signed [BW_PER_ACT + BW_PER_WEIGHT - 1:0] LD_0, LD_1, LD_2, LD_3, LD_4, LD_5, LD_6, LD_7, LD_8, temp_LD_0, temp_LD_1, temp_LD_2, temp_LD_3, temp_LD_4, temp_LD_5, temp_LD_6, temp_LD_7, temp_LD_8;
reg signed [BW_PER_ACT + BW_PER_WEIGHT - 1:0] RD_0, RD_1, RD_2, RD_3, RD_4, RD_5, RD_6, RD_7, RD_8, temp_RD_0, temp_RD_1, temp_RD_2, temp_RD_3, temp_RD_4, temp_RD_5, temp_RD_6, temp_RD_7, temp_RD_8;

//block ff for output
reg signed [BW_PER_ACT + BW_PER_WEIGHT + 8 - 1:0] temp_LU_sum;  //28+8=36-bit
reg signed [BW_PER_ACT + BW_PER_WEIGHT + 8 - 1:0] temp_RU_sum;
reg signed [BW_PER_ACT + BW_PER_WEIGHT + 8 - 1:0] temp_LD_sum;
reg signed [BW_PER_ACT + BW_PER_WEIGHT + 8 - 1:0] temp_RD_sum;


always @*  begin
    w0 =  sram_rdata_weight_delay[base-:BW_PER_WEIGHT];
    w1 =  sram_rdata_weight_delay[(base-BW_PER_WEIGHT)-:BW_PER_WEIGHT];
    w2 =  sram_rdata_weight_delay[(base-2*BW_PER_WEIGHT)-:BW_PER_WEIGHT];
    w3 =  sram_rdata_weight_delay[(base-3*BW_PER_WEIGHT)-:BW_PER_WEIGHT];
    w4 =  sram_rdata_weight_delay[(base-4*BW_PER_WEIGHT)-:BW_PER_WEIGHT];
    w5 =  sram_rdata_weight_delay[(base-5*BW_PER_WEIGHT)-:BW_PER_WEIGHT];
    w6 =  sram_rdata_weight_delay[(base-6*BW_PER_WEIGHT)-:BW_PER_WEIGHT];
    w7 =  sram_rdata_weight_delay[(base-7*BW_PER_WEIGHT)-:BW_PER_WEIGHT];
    w8 =  sram_rdata_weight_delay[(base-8*BW_PER_WEIGHT)-:BW_PER_WEIGHT];
//Left-Up 
    LU_0 = map_0 * w0;
    LU_1 = map_1 * w1;
    LU_2 = map_2 * w2;
    LU_3 = map_4 * w3;
    LU_4 = map_5 * w4;
    LU_5 = map_6 * w5;
    LU_6 = map_8 * w6;
    LU_7 = map_9 * w7;
    LU_8 = map_10 * w8;
//Right-Up
    RU_0 = map_1 * w0;
    RU_1 = map_2 * w1;
    RU_2 = map_3 * w2;
    RU_3 = map_5 * w3;
    RU_4 = map_6 * w4;
    RU_5 = map_7 * w5;
    RU_6 = map_9 * w6;
    RU_7 = map_10 * w7;
    RU_8 = map_11 * w8;
//Left-Down
    LD_0 = map_4 * w0;
    LD_1 = map_5 * w1;
    LD_2 = map_6 * w2;
    LD_3 = map_8 * w3;
    LD_4 = map_9 * w4;
    LD_5 = map_10 * w5;
    LD_6 = map_12 * w6;
    LD_7 = map_13 * w7;
    LD_8 = map_14 * w8;
//Right-Down
    RD_0 = map_5 * w0;
    RD_1 = map_6 * w1;
    RD_2 = map_7 * w2;
    RD_3 = map_9 * w3;
    RD_4 = map_10 * w4;
    RD_5 = map_11 * w5;
    RD_6 = map_13 * w6;
    RD_7 = map_14 * w7;
    RD_8 = map_15 * w8;
//sum
    temp_LU_sum = LU_0 + LU_1 + LU_2 + LU_3 + LU_4 + LU_5 + LU_6 + LU_7 + LU_8;
    temp_RU_sum = RU_0 + RU_1 + RU_2 + RU_3 + RU_4 + RU_5 + RU_6 + RU_7 + RU_8;
    temp_LD_sum = LD_0 + LD_1 + LD_2 + LD_3 + LD_4 + LD_5 + LD_6 + LD_7 + LD_8;
    temp_RD_sum = RD_0 + RD_1 + RD_2 + RD_3 + RD_4 + RD_5 + RD_6 + RD_7 + RD_8;
end

always @(posedge clk)  begin
    if(~rst_n)  begin
        LU_sum <= 0;
        RU_sum <= 0;
        LD_sum <= 0;
        RD_sum <= 0;
    end
    else  begin
        LU_sum <= temp_LU_sum;
        RU_sum <= temp_RU_sum;
        LD_sum <= temp_LD_sum;
        RD_sum <= temp_RD_sum;
    end
end

endmodule