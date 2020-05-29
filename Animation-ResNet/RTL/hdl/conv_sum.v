module conv_sum #(
    parameter CH_NUM = 24,
    parameter ACT_PER_ADDR = 4,
    parameter BW_PER_ACT = 16,
    parameter WEIGHT_PER_ADDR = 216, 
    parameter BIAS_PER_ADDR = 1,
    parameter BW_PER_WEIGHT = 8,
    parameter BW_PER_BIAS = 8
)
(
    input clk,
    input rst_n,
    input [3:0] state,
    input signed [BIAS_PER_ADDR*BW_PER_BIAS-1:0] sram_rdata_bias_delay4,
    input signed [BW_PER_ACT + BW_PER_WEIGHT + 8 - 1:0] ch0,
    input signed [BW_PER_ACT + BW_PER_WEIGHT + 8 - 1:0] ch1,
    input signed [BW_PER_ACT + BW_PER_WEIGHT + 8 - 1:0] ch2,
    input signed [BW_PER_ACT + BW_PER_WEIGHT + 8 - 1:0] ch3,
    input signed [BW_PER_ACT + BW_PER_WEIGHT + 8 - 1:0] ch4,
    input signed [BW_PER_ACT + BW_PER_WEIGHT + 8 - 1:0] ch5,
    input signed [BW_PER_ACT + BW_PER_WEIGHT + 8 - 1:0] ch6,
    input signed [BW_PER_ACT + BW_PER_WEIGHT + 8 - 1:0] ch7,
    input signed [BW_PER_ACT + BW_PER_WEIGHT + 8 - 1:0] ch8,
    input signed [BW_PER_ACT + BW_PER_WEIGHT + 8 - 1:0] ch9,
    input signed [BW_PER_ACT + BW_PER_WEIGHT + 8 - 1:0] ch10,
    input signed [BW_PER_ACT + BW_PER_WEIGHT + 8 - 1:0] ch11,
    input signed [BW_PER_ACT + BW_PER_WEIGHT + 8 - 1:0] ch12,
    input signed [BW_PER_ACT + BW_PER_WEIGHT + 8 - 1:0] ch13,
    input signed [BW_PER_ACT + BW_PER_WEIGHT + 8 - 1:0] ch14,
    input signed [BW_PER_ACT + BW_PER_WEIGHT + 8 - 1:0] ch15,
    input signed [BW_PER_ACT + BW_PER_WEIGHT + 8 - 1:0] ch16,
    input signed [BW_PER_ACT + BW_PER_WEIGHT + 8 - 1:0] ch17,
    input signed [BW_PER_ACT + BW_PER_WEIGHT + 8 - 1:0] ch18,
    input signed [BW_PER_ACT + BW_PER_WEIGHT + 8 - 1:0] ch19,
    input signed [BW_PER_ACT + BW_PER_WEIGHT + 8 - 1:0] ch20,
    input signed [BW_PER_ACT + BW_PER_WEIGHT + 8 - 1:0] ch21,
    input signed [BW_PER_ACT + BW_PER_WEIGHT + 8 - 1:0] ch22,
    input signed [BW_PER_ACT + BW_PER_WEIGHT + 8 - 1:0] ch23,
    input signed [BW_PER_ACT-1:0] forwarding,
    output reg signed [BW_PER_ACT-1:0] pixel_out
);
parameter IDLE = 0, PADDING = 1, CONV1 = 2, RES_1 = 3, RES_2 = 4, UP_1 = 5, UP_2 = 6, CONV2 = 7, FINISH = 8;

reg signed [BW_PER_ACT + BW_PER_WEIGHT + 8 + 11 - 1:0] temp_sum1, sum1;
reg signed [BW_PER_ACT + BW_PER_WEIGHT + 8 + 11 - 1:0] temp_sum2, sum2;
reg signed [BW_PER_ACT + BW_PER_WEIGHT + 8 + 11:0] sum_all;
reg signed [BW_PER_ACT + BW_PER_WEIGHT + 8 + 11:0] sum_round;
reg signed [BW_PER_ACT-1:0] sum_saturate;
reg signed [BW_PER_ACT-1:0] temp_pixel_out;
reg signed [BW_PER_ACT + BW_PER_WEIGHT + 8 + 11:0] sum_test;

always @*  begin
    temp_sum1 = ch0 + ch1 + ch2 + ch3 + ch4 + ch5 + ch6 + ch7 + ch8 + ch9 + ch10 + ch11;
    temp_sum2 = ch12 + ch13 + ch14 + ch15 + ch16 + ch17 + ch18 + ch19 + ch20 + ch21 + ch22 + ch23;

    //shift and plus bias
    if(state==CONV1)  begin
        sum_all =  ((sum1 + sum2) >>> 5) + (  sram_rdata_bias_delay4 <<< 2);//
    end
    else if(state==RES_2)  begin
        sum_all =  ((sum1 + sum2) >>> 6) + ( sram_rdata_bias_delay4 <<< 2) + (forwarding <<< 1);//
    end
    else  begin
        sum_all =  ((sum1 + sum2) >>> 6) + (  sram_rdata_bias_delay4 <<< 2);//
    end

    //rounding
    if(sum_all[0]==1'b1)begin
        sum_test = (1'b1 + sum_all);
        sum_round =  sum_test >>> 1;
    end
    else begin
        sum_round = sum_all >>> 1;
    end

    //saturate
    if(sum_round>(2**(BW_PER_ACT-1)-1))  begin
        sum_saturate = 2**(BW_PER_ACT-1)-1;
    end
    else if(sum_round<-(2**(BW_PER_ACT-1)))  begin
        sum_saturate = -(2**(BW_PER_ACT-1));
    end
    else  begin
        sum_saturate = sum_round[BW_PER_ACT-1:0];
    end

    //relu
    if(state==RES_1)  begin 
        if(sum_saturate<0)
            temp_pixel_out = 0;
        else
            temp_pixel_out = sum_saturate;
    end
    else  begin
        temp_pixel_out = sum_saturate;
    end
end

always @(posedge clk)  begin
    if(~rst_n)  begin
        sum1 <= 0;
        sum2 <= 0;
        pixel_out <= 0;
    end
    else  begin
        sum1 <= temp_sum1;
        sum2 <= temp_sum2;
        pixel_out <= temp_pixel_out;
    end
end

endmodule

