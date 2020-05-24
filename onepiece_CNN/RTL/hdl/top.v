module top #(
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
input clk,                           //clock input
input rst_n,                         //synchronous reset (active low)

input enable,
// input [BW_PER_ACT-1:0] input_data,    //input image(?)

//read data from SRAM group A
input [CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1:0] sram_rdata_a0,    //[1919:0]
input [CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1:0] sram_rdata_a1,
input [CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1:0] sram_rdata_a2,
input [CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1:0] sram_rdata_a3,
//read data from SRAM group B
input [CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1:0] sram_rdata_b0,    //[1919:0]
input [CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1:0] sram_rdata_b1,
input [CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1:0] sram_rdata_b2,
input [CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1:0] sram_rdata_b3,

input [WEIGHT_PER_ADDR*BW_PER_WEIGHT-1:0] sram_rdata_weight,     //read data from SRAM weight  // [1727:0]
input signed [BIAS_PER_ADDR*BW_PER_BIAS-1:0] sram_rdata_bias,    //read data from SRAM bias    // [7:0]

//read address from SRAM group A
output [15:0] sram_raddr_a0,
output [15:0] sram_raddr_a1,
output [15:0] sram_raddr_a2,
output [15:0] sram_raddr_a3,

//read address from SRAM group B
output [15:0] sram_raddr_b0,
output [15:0] sram_raddr_b1,
output [15:0] sram_raddr_b2,
output [15:0] sram_raddr_b3,

output [8:0] sram_raddr_weight,       //read address from SRAM weight  
output [8:0] sram_raddr_bias,         //read address from SRAM bias 

// output     busy, (?)
output reg test_layer_finish,
output reg valid,                         //output valid to check the final answer

//write enable for SRAM groups A & B
output sram_wen_a0,
output sram_wen_a1,
output sram_wen_a2,
output sram_wen_a3,

output sram_wen_b0,
output sram_wen_b1,
output sram_wen_b2,
output sram_wen_b3,

//bytemask for SRAM groups A & B
output [CH_NUM*ACT_PER_ADDR-1:0] sram_bytemask_b0,
output [CH_NUM*ACT_PER_ADDR-1:0] sram_bytemask_b1,
output [CH_NUM*ACT_PER_ADDR-1:0] sram_bytemask_b2,
output [CH_NUM*ACT_PER_ADDR-1:0] sram_bytemask_b3,

//write addrress to SRAM groups A & B
output [15:0] sram_waddr_b0,
output [15:0] sram_waddr_b1,
output [15:0] sram_waddr_b2,
output [15:0] sram_waddr_b3,

//write data to SRAM groups A & B
output [CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1:0] sram_wdata_b0,
output [CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1:0] sram_wdata_b1,
output [CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1:0] sram_wdata_b2,
output [CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1:0] sram_wdata_b3
);

//fsm
parameter IDLE = 0, PADDING = 1, CONV1 = 2, RES_1 = 3, RES_2 = 4, UP_1 = 5, UP_2 = 6, CONV2 = 7, FINISH = 8;
wire [3:0] state;

//row, col ctl
wire [8:0] row;
wire [9:0] col;

//counter
wire [17:0] count;
wire fmap_end;
wire [6:0] fmap_idx_delay4;
wire [6:0] fmap_idx_delay5;
wire output_en;

//sram_read addr ctl
wire [15:0] read_addr0_delay5, read_addr1_delay5, read_addr2_delay5, read_addr3_delay5;

//delay input data
//read data from SRAM group A
reg [CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1:0] sram_rdata_a0_delay;
reg [CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1:0] sram_rdata_a1_delay;
reg [CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1:0] sram_rdata_a2_delay;
reg [CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1:0] sram_rdata_a3_delay;
//read data from SRAM group B
reg [CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1:0] sram_rdata_b0_delay;
reg [CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1:0] sram_rdata_b1_delay;
reg [CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1:0] sram_rdata_b2_delay;
reg [CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1:0] sram_rdata_b3_delay;
//read data from SRAM weight & bias
reg [WEIGHT_PER_ADDR*BW_PER_WEIGHT-1:0] sram_rdata_weight_delay;
reg signed [BIAS_PER_ADDR*BW_PER_BIAS-1:0] sram_rdata_bias_delay, sram_rdata_bias_delay2, sram_rdata_bias_delay3, sram_rdata_bias_delay4;  //delay 4 because pipeline

//determine map type
reg [1:0] map_type, map_type_delay, map_type_delay2, map_type_delay3, map_type_delay4, map_type_delay5; //delay because sram input data are delayed/pipeline

//map generator
wire signed [BW_PER_ACT-1:0] ch0_0, ch0_1, ch0_2, ch0_3, ch0_4, ch0_5, ch0_6, ch0_7, ch0_8, ch0_9, ch0_10, ch0_11, ch0_12, ch0_13, ch0_14, ch0_15;
wire signed [BW_PER_ACT-1:0] ch1_0, ch1_1, ch1_2, ch1_3, ch1_4, ch1_5, ch1_6, ch1_7, ch1_8, ch1_9, ch1_10, ch1_11, ch1_12, ch1_13, ch1_14, ch1_15;
wire signed [BW_PER_ACT-1:0] ch2_0, ch2_1, ch2_2, ch2_3, ch2_4, ch2_5, ch2_6, ch2_7, ch2_8, ch2_9, ch2_10, ch2_11, ch2_12, ch2_13, ch2_14, ch2_15;
wire signed [BW_PER_ACT-1:0] ch3_0, ch3_1, ch3_2, ch3_3, ch3_4, ch3_5, ch3_6, ch3_7, ch3_8, ch3_9, ch3_10, ch3_11, ch3_12, ch3_13, ch3_14, ch3_15;
wire signed [BW_PER_ACT-1:0] ch4_0, ch4_1, ch4_2, ch4_3, ch4_4, ch4_5, ch4_6, ch4_7, ch4_8, ch4_9, ch4_10, ch4_11, ch4_12, ch4_13, ch4_14, ch4_15;
wire signed [BW_PER_ACT-1:0] ch5_0, ch5_1, ch5_2, ch5_3, ch5_4, ch5_5, ch5_6, ch5_7, ch5_8, ch5_9, ch5_10, ch5_11, ch5_12, ch5_13, ch5_14, ch5_15;
wire signed [BW_PER_ACT-1:0] ch6_0, ch6_1, ch6_2, ch6_3, ch6_4, ch6_5, ch6_6, ch6_7, ch6_8, ch6_9, ch6_10, ch6_11, ch6_12, ch6_13, ch6_14, ch6_15;
wire signed [BW_PER_ACT-1:0] ch7_0, ch7_1, ch7_2, ch7_3, ch7_4, ch7_5, ch7_6, ch7_7, ch7_8, ch7_9, ch7_10, ch7_11, ch7_12, ch7_13, ch7_14, ch7_15;
wire signed [BW_PER_ACT-1:0] ch8_0, ch8_1, ch8_2, ch8_3, ch8_4, ch8_5, ch8_6, ch8_7, ch8_8, ch8_9, ch8_10, ch8_11, ch8_12, ch8_13, ch8_14, ch8_15;
wire signed [BW_PER_ACT-1:0] ch9_0, ch9_1, ch9_2, ch9_3, ch9_4, ch9_5, ch9_6, ch9_7, ch9_8, ch9_9, ch9_10, ch9_11, ch9_12, ch9_13, ch9_14, ch9_15;
wire signed [BW_PER_ACT-1:0] ch10_0, ch10_1, ch10_2, ch10_3, ch10_4, ch10_5, ch10_6, ch10_7, ch10_8, ch10_9, ch10_10, ch10_11, ch10_12, ch10_13, ch10_14, ch10_15;
wire signed [BW_PER_ACT-1:0] ch11_0, ch11_1, ch11_2, ch11_3, ch11_4, ch11_5, ch11_6, ch11_7, ch11_8, ch11_9, ch11_10, ch11_11, ch11_12, ch11_13, ch11_14, ch11_15;
wire signed [BW_PER_ACT-1:0] ch12_0, ch12_1, ch12_2, ch12_3, ch12_4, ch12_5, ch12_6, ch12_7, ch12_8, ch12_9, ch12_10, ch12_11, ch12_12, ch12_13, ch12_14, ch12_15;
wire signed [BW_PER_ACT-1:0] ch13_0, ch13_1, ch13_2, ch13_3, ch13_4, ch13_5, ch13_6, ch13_7, ch13_8, ch13_9, ch13_10, ch13_11, ch13_12, ch13_13, ch13_14, ch13_15;
wire signed [BW_PER_ACT-1:0] ch14_0, ch14_1, ch14_2, ch14_3, ch14_4, ch14_5, ch14_6, ch14_7, ch14_8, ch14_9, ch14_10, ch14_11, ch14_12, ch14_13, ch14_14, ch14_15;
wire signed [BW_PER_ACT-1:0] ch15_0, ch15_1, ch15_2, ch15_3, ch15_4, ch15_5, ch15_6, ch15_7, ch15_8, ch15_9, ch15_10, ch15_11, ch15_12, ch15_13, ch15_14, ch15_15;
wire signed [BW_PER_ACT-1:0] ch16_0, ch16_1, ch16_2, ch16_3, ch16_4, ch16_5, ch16_6, ch16_7, ch16_8, ch16_9, ch16_10, ch16_11, ch16_12, ch16_13, ch16_14, ch16_15;
wire signed [BW_PER_ACT-1:0] ch17_0, ch17_1, ch17_2, ch17_3, ch17_4, ch17_5, ch17_6, ch17_7, ch17_8, ch17_9, ch17_10, ch17_11, ch17_12, ch17_13, ch17_14, ch17_15;
wire signed [BW_PER_ACT-1:0] ch18_0, ch18_1, ch18_2, ch18_3, ch18_4, ch18_5, ch18_6, ch18_7, ch18_8, ch18_9, ch18_10, ch18_11, ch18_12, ch18_13, ch18_14, ch18_15;
wire signed [BW_PER_ACT-1:0] ch19_0, ch19_1, ch19_2, ch19_3, ch19_4, ch19_5, ch19_6, ch19_7, ch19_8, ch19_9, ch19_10, ch19_11, ch19_12, ch19_13, ch19_14, ch19_15;
wire signed [BW_PER_ACT-1:0] ch20_0, ch20_1, ch20_2, ch20_3, ch20_4, ch20_5, ch20_6, ch20_7, ch20_8, ch20_9, ch20_10, ch20_11, ch20_12, ch20_13, ch20_14, ch20_15;
wire signed [BW_PER_ACT-1:0] ch21_0, ch21_1, ch21_2, ch21_3, ch21_4, ch21_5, ch21_6, ch21_7, ch21_8, ch21_9, ch21_10, ch21_11, ch21_12, ch21_13, ch21_14, ch21_15;
wire signed [BW_PER_ACT-1:0] ch22_0, ch22_1, ch22_2, ch22_3, ch22_4, ch22_5, ch22_6, ch22_7, ch22_8, ch22_9, ch22_10, ch22_11, ch22_12, ch22_13, ch22_14, ch22_15;
wire signed [BW_PER_ACT-1:0] ch23_0, ch23_1, ch23_2, ch23_3, ch23_4, ch23_5, ch23_6, ch23_7, ch23_8, ch23_9, ch23_10, ch23_11, ch23_12, ch23_13, ch23_14, ch23_15;

//conv_mul
wire signed [BW_PER_ACT + BW_PER_WEIGHT + 8 - 1:0] ch0_LU_sum, ch0_RU_sum, ch0_LD_sum, ch0_RD_sum;
wire signed [BW_PER_ACT + BW_PER_WEIGHT + 8 - 1:0] ch1_LU_sum, ch1_RU_sum, ch1_LD_sum, ch1_RD_sum;
wire signed [BW_PER_ACT + BW_PER_WEIGHT + 8 - 1:0] ch2_LU_sum, ch2_RU_sum, ch2_LD_sum, ch2_RD_sum;
wire signed [BW_PER_ACT + BW_PER_WEIGHT + 8 - 1:0] ch3_LU_sum, ch3_RU_sum, ch3_LD_sum, ch3_RD_sum;
wire signed [BW_PER_ACT + BW_PER_WEIGHT + 8 - 1:0] ch4_LU_sum, ch4_RU_sum, ch4_LD_sum, ch4_RD_sum;
wire signed [BW_PER_ACT + BW_PER_WEIGHT + 8 - 1:0] ch5_LU_sum, ch5_RU_sum, ch5_LD_sum, ch5_RD_sum;
wire signed [BW_PER_ACT + BW_PER_WEIGHT + 8 - 1:0] ch6_LU_sum, ch6_RU_sum, ch6_LD_sum, ch6_RD_sum;
wire signed [BW_PER_ACT + BW_PER_WEIGHT + 8 - 1:0] ch7_LU_sum, ch7_RU_sum, ch7_LD_sum, ch7_RD_sum;
wire signed [BW_PER_ACT + BW_PER_WEIGHT + 8 - 1:0] ch8_LU_sum, ch8_RU_sum, ch8_LD_sum, ch8_RD_sum;
wire signed [BW_PER_ACT + BW_PER_WEIGHT + 8 - 1:0] ch9_LU_sum, ch9_RU_sum, ch9_LD_sum, ch9_RD_sum;
wire signed [BW_PER_ACT + BW_PER_WEIGHT + 8 - 1:0] ch10_LU_sum, ch10_RU_sum, ch10_LD_sum, ch10_RD_sum;
wire signed [BW_PER_ACT + BW_PER_WEIGHT + 8 - 1:0] ch11_LU_sum, ch11_RU_sum, ch11_LD_sum, ch11_RD_sum;
wire signed [BW_PER_ACT + BW_PER_WEIGHT + 8 - 1:0] ch12_LU_sum, ch12_RU_sum, ch12_LD_sum, ch12_RD_sum;
wire signed [BW_PER_ACT + BW_PER_WEIGHT + 8 - 1:0] ch13_LU_sum, ch13_RU_sum, ch13_LD_sum, ch13_RD_sum;
wire signed [BW_PER_ACT + BW_PER_WEIGHT + 8 - 1:0] ch14_LU_sum, ch14_RU_sum, ch14_LD_sum, ch14_RD_sum;
wire signed [BW_PER_ACT + BW_PER_WEIGHT + 8 - 1:0] ch15_LU_sum, ch15_RU_sum, ch15_LD_sum, ch15_RD_sum;
wire signed [BW_PER_ACT + BW_PER_WEIGHT + 8 - 1:0] ch16_LU_sum, ch16_RU_sum, ch16_LD_sum, ch16_RD_sum;
wire signed [BW_PER_ACT + BW_PER_WEIGHT + 8 - 1:0] ch17_LU_sum, ch17_RU_sum, ch17_LD_sum, ch17_RD_sum;
wire signed [BW_PER_ACT + BW_PER_WEIGHT + 8 - 1:0] ch18_LU_sum, ch18_RU_sum, ch18_LD_sum, ch18_RD_sum;
wire signed [BW_PER_ACT + BW_PER_WEIGHT + 8 - 1:0] ch19_LU_sum, ch19_RU_sum, ch19_LD_sum, ch19_RD_sum;
wire signed [BW_PER_ACT + BW_PER_WEIGHT + 8 - 1:0] ch20_LU_sum, ch20_RU_sum, ch20_LD_sum, ch20_RD_sum;
wire signed [BW_PER_ACT + BW_PER_WEIGHT + 8 - 1:0] ch21_LU_sum, ch21_RU_sum, ch21_LD_sum, ch21_RD_sum;
wire signed [BW_PER_ACT + BW_PER_WEIGHT + 8 - 1:0] ch22_LU_sum, ch22_RU_sum, ch22_LD_sum, ch22_RD_sum;
wire signed [BW_PER_ACT + BW_PER_WEIGHT + 8 - 1:0] ch23_LU_sum, ch23_RU_sum, ch23_LD_sum, ch23_RD_sum;

//conv_sum
wire signed [BW_PER_ACT-1:0] LU_out, RU_out, LD_out, RD_out;

//sramB_write_ctl
wire pad_end;

//resblock_forwarding
wire signed [BW_PER_ACT-1:0] LU_forwarding, RU_forwarding, LD_forwarding, RD_forwarding;

//map_base
reg [BASE_BW-1:0] base [0:95];
reg [BASE_BW-1:0] base_w [0:23];
integer i;

//FINISH
always @*  begin
    //map_generator base calculation
    for(i=0; i<96; i=i+1)  begin
        base[i] =  (CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1) - (i*BW_PER_ACT);
    end
    //conv_mul base calculation
    for(i=0; i<24; i=i+1)  begin
        base_w[i] =  (CH_NUM*9*BW_PER_WEIGHT-1) - (i*9*BW_PER_WEIGHT);
    end
    //finish 
    // MODIFY: (current: Conv1)
    // If compare CONV1: RES_1
    // If compare RES_1: RES_2
    if(state==FINISH)  begin
        test_layer_finish = 1;
    end
    else  begin
        test_layer_finish = 0;
    end
end

//--------module calling--------
fsm fsm(
    .clk(clk),
    .rst_n(rst_n),
    .enable(enable),
    .fmap_idx_delay4(fmap_idx_delay4),
    .pad_end(pad_end),
    .state(state)
);

row_col_ctl row_col_ctl(
    .clk(clk),
    .rst_n(rst_n),
    .state(state),
    .fmap_end(fmap_end),
    .fmap_idx_delay4(fmap_idx_delay4),
    .row(row),
    .col(col)
);

counter counter(
    .clk(clk),
    .rst_n(rst_n),
    .state(state),
    .count(count),
    .fmap_end(fmap_end),
    .fmap_idx_delay4(fmap_idx_delay4),
    .fmap_idx_delay5(fmap_idx_delay5),
    .output_en(output_en)
);

sram_read_addr_ctl sram_read_addr_ctl(
    .clk(clk),
    .rst_n(rst_n),
    .row(row),
    .col(col),
    .state(state),
    .fmap_end(fmap_end),
    .fmap_idx_delay4(fmap_idx_delay4),
    .sram_raddr_a0(sram_raddr_a0),
    .sram_raddr_a1(sram_raddr_a1),
    .sram_raddr_a2(sram_raddr_a2),
    .sram_raddr_a3(sram_raddr_a3),
    .sram_raddr_b0(sram_raddr_b0),
    .sram_raddr_b1(sram_raddr_b1),
    .sram_raddr_b2(sram_raddr_b2),
    .sram_raddr_b3(sram_raddr_b3),
    .sram_raddr_weight(sram_raddr_weight),
    .sram_raddr_bias(sram_raddr_bias),
    .read_addr0_delay5(read_addr0_delay5),
    .read_addr1_delay5(read_addr1_delay5),
    .read_addr2_delay5(read_addr2_delay5),
    .read_addr3_delay5(read_addr3_delay5)
);

//map generator
map_generator map0(
    .map_type_delay(map_type_delay),
    .state(state),
    .sram_rdata_a0_delay(sram_rdata_a0_delay),
    .sram_rdata_a1_delay(sram_rdata_a1_delay),
    .sram_rdata_a2_delay(sram_rdata_a2_delay),
    .sram_rdata_a3_delay(sram_rdata_a3_delay),
    .sram_rdata_b0_delay(sram_rdata_b0_delay),
    .sram_rdata_b1_delay(sram_rdata_b1_delay),
    .sram_rdata_b2_delay(sram_rdata_b2_delay),
    .sram_rdata_b3_delay(sram_rdata_b3_delay),
    .base_0(base[0]),
    .base_1(base[1]),
    .base_2(base[2]),
    .base_3(base[3]),
    .map_0(ch0_0),
    .map_1(ch0_1),
    .map_2(ch0_2),
    .map_3(ch0_3),
    .map_4(ch0_4),
    .map_5(ch0_5),
    .map_6(ch0_6),
    .map_7(ch0_7),
    .map_8(ch0_8),
    .map_9(ch0_9),
    .map_10(ch0_10),
    .map_11(ch0_11),
    .map_12(ch0_12),
    .map_13(ch0_13),
    .map_14(ch0_14),
    .map_15(ch0_15)
);
map_generator map1(
    .map_type_delay(map_type_delay),
    .state(state),
    .sram_rdata_a0_delay(sram_rdata_a0_delay),
    .sram_rdata_a1_delay(sram_rdata_a1_delay),
    .sram_rdata_a2_delay(sram_rdata_a2_delay),
    .sram_rdata_a3_delay(sram_rdata_a3_delay),
    .sram_rdata_b0_delay(sram_rdata_b0_delay),
    .sram_rdata_b1_delay(sram_rdata_b1_delay),
    .sram_rdata_b2_delay(sram_rdata_b2_delay),
    .sram_rdata_b3_delay(sram_rdata_b3_delay),
    .base_0(base[4]),
    .base_1(base[5]),
    .base_2(base[6]),
    .base_3(base[7]),
    .map_0(ch1_0),
    .map_1(ch1_1),
    .map_2(ch1_2),
    .map_3(ch1_3),
    .map_4(ch1_4),
    .map_5(ch1_5),
    .map_6(ch1_6),
    .map_7(ch1_7),
    .map_8(ch1_8),
    .map_9(ch1_9),
    .map_10(ch1_10),
    .map_11(ch1_11),
    .map_12(ch1_12),
    .map_13(ch1_13),
    .map_14(ch1_14),
    .map_15(ch1_15)
);
map_generator map2(
    .map_type_delay(map_type_delay),
    .state(state),
    .sram_rdata_a0_delay(sram_rdata_a0_delay),
    .sram_rdata_a1_delay(sram_rdata_a1_delay),
    .sram_rdata_a2_delay(sram_rdata_a2_delay),
    .sram_rdata_a3_delay(sram_rdata_a3_delay),
    .sram_rdata_b0_delay(sram_rdata_b0_delay),
    .sram_rdata_b1_delay(sram_rdata_b1_delay),
    .sram_rdata_b2_delay(sram_rdata_b2_delay),
    .sram_rdata_b3_delay(sram_rdata_b3_delay),
    .base_0(base[8]),
    .base_1(base[9]),
    .base_2(base[10]),
    .base_3(base[11]),
    .map_0(ch2_0),
    .map_1(ch2_1),
    .map_2(ch2_2),
    .map_3(ch2_3),
    .map_4(ch2_4),
    .map_5(ch2_5),
    .map_6(ch2_6),
    .map_7(ch2_7),
    .map_8(ch2_8),
    .map_9(ch2_9),
    .map_10(ch2_10),
    .map_11(ch2_11),
    .map_12(ch2_12),
    .map_13(ch2_13),
    .map_14(ch2_14),
    .map_15(ch2_15)
);
map_generator map3(
    .map_type_delay(map_type_delay),
    .state(state),
    .sram_rdata_a0_delay(sram_rdata_a0_delay),
    .sram_rdata_a1_delay(sram_rdata_a1_delay),
    .sram_rdata_a2_delay(sram_rdata_a2_delay),
    .sram_rdata_a3_delay(sram_rdata_a3_delay),
    .sram_rdata_b0_delay(sram_rdata_b0_delay),
    .sram_rdata_b1_delay(sram_rdata_b1_delay),
    .sram_rdata_b2_delay(sram_rdata_b2_delay),
    .sram_rdata_b3_delay(sram_rdata_b3_delay),
    .base_0(base[12]),
    .base_1(base[13]),
    .base_2(base[14]),
    .base_3(base[15]),
    .map_0(ch3_0),
    .map_1(ch3_1),
    .map_2(ch3_2),
    .map_3(ch3_3),
    .map_4(ch3_4),
    .map_5(ch3_5),
    .map_6(ch3_6),
    .map_7(ch3_7),
    .map_8(ch3_8),
    .map_9(ch3_9),
    .map_10(ch3_10),
    .map_11(ch3_11),
    .map_12(ch3_12),
    .map_13(ch3_13),
    .map_14(ch3_14),
    .map_15(ch3_15)
);
map_generator map4(
    .map_type_delay(map_type_delay),
    .state(state),
    .sram_rdata_a0_delay(sram_rdata_a0_delay),
    .sram_rdata_a1_delay(sram_rdata_a1_delay),
    .sram_rdata_a2_delay(sram_rdata_a2_delay),
    .sram_rdata_a3_delay(sram_rdata_a3_delay),
    .sram_rdata_b0_delay(sram_rdata_b0_delay),
    .sram_rdata_b1_delay(sram_rdata_b1_delay),
    .sram_rdata_b2_delay(sram_rdata_b2_delay),
    .sram_rdata_b3_delay(sram_rdata_b3_delay),
    .base_0(base[16]),
    .base_1(base[17]),
    .base_2(base[18]),
    .base_3(base[19]),
    .map_0(ch4_0),
    .map_1(ch4_1),
    .map_2(ch4_2),
    .map_3(ch4_3),
    .map_4(ch4_4),
    .map_5(ch4_5),
    .map_6(ch4_6),
    .map_7(ch4_7),
    .map_8(ch4_8),
    .map_9(ch4_9),
    .map_10(ch4_10),
    .map_11(ch4_11),
    .map_12(ch4_12),
    .map_13(ch4_13),
    .map_14(ch4_14),
    .map_15(ch4_15)
);
map_generator map5(
    .map_type_delay(map_type_delay),
    .state(state),
    .sram_rdata_a0_delay(sram_rdata_a0_delay),
    .sram_rdata_a1_delay(sram_rdata_a1_delay),
    .sram_rdata_a2_delay(sram_rdata_a2_delay),
    .sram_rdata_a3_delay(sram_rdata_a3_delay),
    .sram_rdata_b0_delay(sram_rdata_b0_delay),
    .sram_rdata_b1_delay(sram_rdata_b1_delay),
    .sram_rdata_b2_delay(sram_rdata_b2_delay),
    .sram_rdata_b3_delay(sram_rdata_b3_delay),
    .base_0(base[20]),
    .base_1(base[21]),
    .base_2(base[22]),
    .base_3(base[23]),
    .map_0(ch5_0),
    .map_1(ch5_1),
    .map_2(ch5_2),
    .map_3(ch5_3),
    .map_4(ch5_4),
    .map_5(ch5_5),
    .map_6(ch5_6),
    .map_7(ch5_7),
    .map_8(ch5_8),
    .map_9(ch5_9),
    .map_10(ch5_10),
    .map_11(ch5_11),
    .map_12(ch5_12),
    .map_13(ch5_13),
    .map_14(ch5_14),
    .map_15(ch5_15)
);
map_generator map6(
    .map_type_delay(map_type_delay),
    .state(state),
    .sram_rdata_a0_delay(sram_rdata_a0_delay),
    .sram_rdata_a1_delay(sram_rdata_a1_delay),
    .sram_rdata_a2_delay(sram_rdata_a2_delay),
    .sram_rdata_a3_delay(sram_rdata_a3_delay),
    .sram_rdata_b0_delay(sram_rdata_b0_delay),
    .sram_rdata_b1_delay(sram_rdata_b1_delay),
    .sram_rdata_b2_delay(sram_rdata_b2_delay),
    .sram_rdata_b3_delay(sram_rdata_b3_delay),
    .base_0(base[24]),
    .base_1(base[25]),
    .base_2(base[26]),
    .base_3(base[27]),
    .map_0(ch6_0),
    .map_1(ch6_1),
    .map_2(ch6_2),
    .map_3(ch6_3),
    .map_4(ch6_4),
    .map_5(ch6_5),
    .map_6(ch6_6),
    .map_7(ch6_7),
    .map_8(ch6_8),
    .map_9(ch6_9),
    .map_10(ch6_10),
    .map_11(ch6_11),
    .map_12(ch6_12),
    .map_13(ch6_13),
    .map_14(ch6_14),
    .map_15(ch6_15)
);
map_generator map7(
    .map_type_delay(map_type_delay),
    .state(state),
    .sram_rdata_a0_delay(sram_rdata_a0_delay),
    .sram_rdata_a1_delay(sram_rdata_a1_delay),
    .sram_rdata_a2_delay(sram_rdata_a2_delay),
    .sram_rdata_a3_delay(sram_rdata_a3_delay),
    .sram_rdata_b0_delay(sram_rdata_b0_delay),
    .sram_rdata_b1_delay(sram_rdata_b1_delay),
    .sram_rdata_b2_delay(sram_rdata_b2_delay),
    .sram_rdata_b3_delay(sram_rdata_b3_delay),
    .base_0(base[28]),
    .base_1(base[29]),
    .base_2(base[30]),
    .base_3(base[31]),
    .map_0(ch7_0),
    .map_1(ch7_1),
    .map_2(ch7_2),
    .map_3(ch7_3),
    .map_4(ch7_4),
    .map_5(ch7_5),
    .map_6(ch7_6),
    .map_7(ch7_7),
    .map_8(ch7_8),
    .map_9(ch7_9),
    .map_10(ch7_10),
    .map_11(ch7_11),
    .map_12(ch7_12),
    .map_13(ch7_13),
    .map_14(ch7_14),
    .map_15(ch7_15)
);
map_generator map8(
    .map_type_delay(map_type_delay),
    .state(state),
    .sram_rdata_a0_delay(sram_rdata_a0_delay),
    .sram_rdata_a1_delay(sram_rdata_a1_delay),
    .sram_rdata_a2_delay(sram_rdata_a2_delay),
    .sram_rdata_a3_delay(sram_rdata_a3_delay),
    .sram_rdata_b0_delay(sram_rdata_b0_delay),
    .sram_rdata_b1_delay(sram_rdata_b1_delay),
    .sram_rdata_b2_delay(sram_rdata_b2_delay),
    .sram_rdata_b3_delay(sram_rdata_b3_delay),
    .base_0(base[32]),
    .base_1(base[33]),
    .base_2(base[34]),
    .base_3(base[35]),
    .map_0(ch8_0),
    .map_1(ch8_1),
    .map_2(ch8_2),
    .map_3(ch8_3),
    .map_4(ch8_4),
    .map_5(ch8_5),
    .map_6(ch8_6),
    .map_7(ch8_7),
    .map_8(ch8_8),
    .map_9(ch8_9),
    .map_10(ch8_10),
    .map_11(ch8_11),
    .map_12(ch8_12),
    .map_13(ch8_13),
    .map_14(ch8_14),
    .map_15(ch8_15)
);
map_generator map9(
    .map_type_delay(map_type_delay),
    .state(state),
    .sram_rdata_a0_delay(sram_rdata_a0_delay),
    .sram_rdata_a1_delay(sram_rdata_a1_delay),
    .sram_rdata_a2_delay(sram_rdata_a2_delay),
    .sram_rdata_a3_delay(sram_rdata_a3_delay),
    .sram_rdata_b0_delay(sram_rdata_b0_delay),
    .sram_rdata_b1_delay(sram_rdata_b1_delay),
    .sram_rdata_b2_delay(sram_rdata_b2_delay),
    .sram_rdata_b3_delay(sram_rdata_b3_delay),
    .base_0(base[36]),
    .base_1(base[37]),
    .base_2(base[38]),
    .base_3(base[39]),
    .map_0(ch9_0),
    .map_1(ch9_1),
    .map_2(ch9_2),
    .map_3(ch9_3),
    .map_4(ch9_4),
    .map_5(ch9_5),
    .map_6(ch9_6),
    .map_7(ch9_7),
    .map_8(ch9_8),
    .map_9(ch9_9),
    .map_10(ch9_10),
    .map_11(ch9_11),
    .map_12(ch9_12),
    .map_13(ch9_13),
    .map_14(ch9_14),
    .map_15(ch9_15)
);
map_generator map10(
    .map_type_delay(map_type_delay),
    .state(state),
    .sram_rdata_a0_delay(sram_rdata_a0_delay),
    .sram_rdata_a1_delay(sram_rdata_a1_delay),
    .sram_rdata_a2_delay(sram_rdata_a2_delay),
    .sram_rdata_a3_delay(sram_rdata_a3_delay),
    .sram_rdata_b0_delay(sram_rdata_b0_delay),
    .sram_rdata_b1_delay(sram_rdata_b1_delay),
    .sram_rdata_b2_delay(sram_rdata_b2_delay),
    .sram_rdata_b3_delay(sram_rdata_b3_delay),
    .base_0(base[40]),
    .base_1(base[41]),
    .base_2(base[42]),
    .base_3(base[43]),
    .map_0(ch10_0),
    .map_1(ch10_1),
    .map_2(ch10_2),
    .map_3(ch10_3),
    .map_4(ch10_4),
    .map_5(ch10_5),
    .map_6(ch10_6),
    .map_7(ch10_7),
    .map_8(ch10_8),
    .map_9(ch10_9),
    .map_10(ch10_10),
    .map_11(ch10_11),
    .map_12(ch10_12),
    .map_13(ch10_13),
    .map_14(ch10_14),
    .map_15(ch10_15)
);
map_generator map11(
    .map_type_delay(map_type_delay),
    .state(state),
    .sram_rdata_a0_delay(sram_rdata_a0_delay),
    .sram_rdata_a1_delay(sram_rdata_a1_delay),
    .sram_rdata_a2_delay(sram_rdata_a2_delay),
    .sram_rdata_a3_delay(sram_rdata_a3_delay),
    .sram_rdata_b0_delay(sram_rdata_b0_delay),
    .sram_rdata_b1_delay(sram_rdata_b1_delay),
    .sram_rdata_b2_delay(sram_rdata_b2_delay),
    .sram_rdata_b3_delay(sram_rdata_b3_delay),
    .base_0(base[44]),
    .base_1(base[45]),
    .base_2(base[46]),
    .base_3(base[47]),
    .map_0(ch11_0),
    .map_1(ch11_1),
    .map_2(ch11_2),
    .map_3(ch11_3),
    .map_4(ch11_4),
    .map_5(ch11_5),
    .map_6(ch11_6),
    .map_7(ch11_7),
    .map_8(ch11_8),
    .map_9(ch11_9),
    .map_10(ch11_10),
    .map_11(ch11_11),
    .map_12(ch11_12),
    .map_13(ch11_13),
    .map_14(ch11_14),
    .map_15(ch11_15)
);
map_generator map12(
    .map_type_delay(map_type_delay),
    .state(state),
    .sram_rdata_a0_delay(sram_rdata_a0_delay),
    .sram_rdata_a1_delay(sram_rdata_a1_delay),
    .sram_rdata_a2_delay(sram_rdata_a2_delay),
    .sram_rdata_a3_delay(sram_rdata_a3_delay),
    .sram_rdata_b0_delay(sram_rdata_b0_delay),
    .sram_rdata_b1_delay(sram_rdata_b1_delay),
    .sram_rdata_b2_delay(sram_rdata_b2_delay),
    .sram_rdata_b3_delay(sram_rdata_b3_delay),
    .base_0(base[48]),
    .base_1(base[49]),
    .base_2(base[50]),
    .base_3(base[51]),
    .map_0(ch12_0),
    .map_1(ch12_1),
    .map_2(ch12_2),
    .map_3(ch12_3),
    .map_4(ch12_4),
    .map_5(ch12_5),
    .map_6(ch12_6),
    .map_7(ch12_7),
    .map_8(ch12_8),
    .map_9(ch12_9),
    .map_10(ch12_10),
    .map_11(ch12_11),
    .map_12(ch12_12),
    .map_13(ch12_13),
    .map_14(ch12_14),
    .map_15(ch12_15)
);
map_generator map13(
    .map_type_delay(map_type_delay),
    .state(state),
    .sram_rdata_a0_delay(sram_rdata_a0_delay),
    .sram_rdata_a1_delay(sram_rdata_a1_delay),
    .sram_rdata_a2_delay(sram_rdata_a2_delay),
    .sram_rdata_a3_delay(sram_rdata_a3_delay),
    .sram_rdata_b0_delay(sram_rdata_b0_delay),
    .sram_rdata_b1_delay(sram_rdata_b1_delay),
    .sram_rdata_b2_delay(sram_rdata_b2_delay),
    .sram_rdata_b3_delay(sram_rdata_b3_delay),
    .base_0(base[52]),
    .base_1(base[53]),
    .base_2(base[54]),
    .base_3(base[55]),
    .map_0(ch13_0),
    .map_1(ch13_1),
    .map_2(ch13_2),
    .map_3(ch13_3),
    .map_4(ch13_4),
    .map_5(ch13_5),
    .map_6(ch13_6),
    .map_7(ch13_7),
    .map_8(ch13_8),
    .map_9(ch13_9),
    .map_10(ch13_10),
    .map_11(ch13_11),
    .map_12(ch13_12),
    .map_13(ch13_13),
    .map_14(ch13_14),
    .map_15(ch13_15)
);
map_generator map14(
    .map_type_delay(map_type_delay),
    .state(state),
    .sram_rdata_a0_delay(sram_rdata_a0_delay),
    .sram_rdata_a1_delay(sram_rdata_a1_delay),
    .sram_rdata_a2_delay(sram_rdata_a2_delay),
    .sram_rdata_a3_delay(sram_rdata_a3_delay),
    .sram_rdata_b0_delay(sram_rdata_b0_delay),
    .sram_rdata_b1_delay(sram_rdata_b1_delay),
    .sram_rdata_b2_delay(sram_rdata_b2_delay),
    .sram_rdata_b3_delay(sram_rdata_b3_delay),
    .base_0(base[56]),
    .base_1(base[57]),
    .base_2(base[58]),
    .base_3(base[59]),
    .map_0(ch14_0),
    .map_1(ch14_1),
    .map_2(ch14_2),
    .map_3(ch14_3),
    .map_4(ch14_4),
    .map_5(ch14_5),
    .map_6(ch14_6),
    .map_7(ch14_7),
    .map_8(ch14_8),
    .map_9(ch14_9),
    .map_10(ch14_10),
    .map_11(ch14_11),
    .map_12(ch14_12),
    .map_13(ch14_13),
    .map_14(ch14_14),
    .map_15(ch14_15)
);
map_generator map15(
    .map_type_delay(map_type_delay),
    .state(state),
    .sram_rdata_a0_delay(sram_rdata_a0_delay),
    .sram_rdata_a1_delay(sram_rdata_a1_delay),
    .sram_rdata_a2_delay(sram_rdata_a2_delay),
    .sram_rdata_a3_delay(sram_rdata_a3_delay),
    .sram_rdata_b0_delay(sram_rdata_b0_delay),
    .sram_rdata_b1_delay(sram_rdata_b1_delay),
    .sram_rdata_b2_delay(sram_rdata_b2_delay),
    .sram_rdata_b3_delay(sram_rdata_b3_delay),
    .base_0(base[60]),
    .base_1(base[61]),
    .base_2(base[62]),
    .base_3(base[63]),
    .map_0(ch15_0),
    .map_1(ch15_1),
    .map_2(ch15_2),
    .map_3(ch15_3),
    .map_4(ch15_4),
    .map_5(ch15_5),
    .map_6(ch15_6),
    .map_7(ch15_7),
    .map_8(ch15_8),
    .map_9(ch15_9),
    .map_10(ch15_10),
    .map_11(ch15_11),
    .map_12(ch15_12),
    .map_13(ch15_13),
    .map_14(ch15_14),
    .map_15(ch15_15)
);
map_generator map16(
    .map_type_delay(map_type_delay),
    .state(state),
    .sram_rdata_a0_delay(sram_rdata_a0_delay),
    .sram_rdata_a1_delay(sram_rdata_a1_delay),
    .sram_rdata_a2_delay(sram_rdata_a2_delay),
    .sram_rdata_a3_delay(sram_rdata_a3_delay),
    .sram_rdata_b0_delay(sram_rdata_b0_delay),
    .sram_rdata_b1_delay(sram_rdata_b1_delay),
    .sram_rdata_b2_delay(sram_rdata_b2_delay),
    .sram_rdata_b3_delay(sram_rdata_b3_delay),
    .base_0(base[64]),
    .base_1(base[65]),
    .base_2(base[66]),
    .base_3(base[67]),
    .map_0(ch16_0),
    .map_1(ch16_1),
    .map_2(ch16_2),
    .map_3(ch16_3),
    .map_4(ch16_4),
    .map_5(ch16_5),
    .map_6(ch16_6),
    .map_7(ch16_7),
    .map_8(ch16_8),
    .map_9(ch16_9),
    .map_10(ch16_10),
    .map_11(ch16_11),
    .map_12(ch16_12),
    .map_13(ch16_13),
    .map_14(ch16_14),
    .map_15(ch16_15)
);
map_generator map17(
    .map_type_delay(map_type_delay),
    .state(state),
    .sram_rdata_a0_delay(sram_rdata_a0_delay),
    .sram_rdata_a1_delay(sram_rdata_a1_delay),
    .sram_rdata_a2_delay(sram_rdata_a2_delay),
    .sram_rdata_a3_delay(sram_rdata_a3_delay),
    .sram_rdata_b0_delay(sram_rdata_b0_delay),
    .sram_rdata_b1_delay(sram_rdata_b1_delay),
    .sram_rdata_b2_delay(sram_rdata_b2_delay),
    .sram_rdata_b3_delay(sram_rdata_b3_delay),
    .base_0(base[68]),
    .base_1(base[69]),
    .base_2(base[70]),
    .base_3(base[71]),
    .map_0(ch17_0),
    .map_1(ch17_1),
    .map_2(ch17_2),
    .map_3(ch17_3),
    .map_4(ch17_4),
    .map_5(ch17_5),
    .map_6(ch17_6),
    .map_7(ch17_7),
    .map_8(ch17_8),
    .map_9(ch17_9),
    .map_10(ch17_10),
    .map_11(ch17_11),
    .map_12(ch17_12),
    .map_13(ch17_13),
    .map_14(ch17_14),
    .map_15(ch17_15)
);
map_generator map18(
    .map_type_delay(map_type_delay),
    .state(state),
    .sram_rdata_a0_delay(sram_rdata_a0_delay),
    .sram_rdata_a1_delay(sram_rdata_a1_delay),
    .sram_rdata_a2_delay(sram_rdata_a2_delay),
    .sram_rdata_a3_delay(sram_rdata_a3_delay),
    .sram_rdata_b0_delay(sram_rdata_b0_delay),
    .sram_rdata_b1_delay(sram_rdata_b1_delay),
    .sram_rdata_b2_delay(sram_rdata_b2_delay),
    .sram_rdata_b3_delay(sram_rdata_b3_delay),
    .base_0(base[72]),
    .base_1(base[73]),
    .base_2(base[74]),
    .base_3(base[75]),
    .map_0(ch18_0),
    .map_1(ch18_1),
    .map_2(ch18_2),
    .map_3(ch18_3),
    .map_4(ch18_4),
    .map_5(ch18_5),
    .map_6(ch18_6),
    .map_7(ch18_7),
    .map_8(ch18_8),
    .map_9(ch18_9),
    .map_10(ch18_10),
    .map_11(ch18_11),
    .map_12(ch18_12),
    .map_13(ch18_13),
    .map_14(ch18_14),
    .map_15(ch18_15)
);
map_generator map19(
    .map_type_delay(map_type_delay),
    .state(state),
    .sram_rdata_a0_delay(sram_rdata_a0_delay),
    .sram_rdata_a1_delay(sram_rdata_a1_delay),
    .sram_rdata_a2_delay(sram_rdata_a2_delay),
    .sram_rdata_a3_delay(sram_rdata_a3_delay),
    .sram_rdata_b0_delay(sram_rdata_b0_delay),
    .sram_rdata_b1_delay(sram_rdata_b1_delay),
    .sram_rdata_b2_delay(sram_rdata_b2_delay),
    .sram_rdata_b3_delay(sram_rdata_b3_delay),
    .base_0(base[76]),
    .base_1(base[77]),
    .base_2(base[78]),
    .base_3(base[79]),
    .map_0(ch19_0),
    .map_1(ch19_1),
    .map_2(ch19_2),
    .map_3(ch19_3),
    .map_4(ch19_4),
    .map_5(ch19_5),
    .map_6(ch19_6),
    .map_7(ch19_7),
    .map_8(ch19_8),
    .map_9(ch19_9),
    .map_10(ch19_10),
    .map_11(ch19_11),
    .map_12(ch19_12),
    .map_13(ch19_13),
    .map_14(ch19_14),
    .map_15(ch19_15)
);
map_generator map20(
    .map_type_delay(map_type_delay),
    .state(state),
    .sram_rdata_a0_delay(sram_rdata_a0_delay),
    .sram_rdata_a1_delay(sram_rdata_a1_delay),
    .sram_rdata_a2_delay(sram_rdata_a2_delay),
    .sram_rdata_a3_delay(sram_rdata_a3_delay),
    .sram_rdata_b0_delay(sram_rdata_b0_delay),
    .sram_rdata_b1_delay(sram_rdata_b1_delay),
    .sram_rdata_b2_delay(sram_rdata_b2_delay),
    .sram_rdata_b3_delay(sram_rdata_b3_delay),
    .base_0(base[80]),
    .base_1(base[81]),
    .base_2(base[82]),
    .base_3(base[83]),
    .map_0(ch20_0),
    .map_1(ch20_1),
    .map_2(ch20_2),
    .map_3(ch20_3),
    .map_4(ch20_4),
    .map_5(ch20_5),
    .map_6(ch20_6),
    .map_7(ch20_7),
    .map_8(ch20_8),
    .map_9(ch20_9),
    .map_10(ch20_10),
    .map_11(ch20_11),
    .map_12(ch20_12),
    .map_13(ch20_13),
    .map_14(ch20_14),
    .map_15(ch20_15)
);
map_generator map21(
    .map_type_delay(map_type_delay),
    .state(state),
    .sram_rdata_a0_delay(sram_rdata_a0_delay),
    .sram_rdata_a1_delay(sram_rdata_a1_delay),
    .sram_rdata_a2_delay(sram_rdata_a2_delay),
    .sram_rdata_a3_delay(sram_rdata_a3_delay),
    .sram_rdata_b0_delay(sram_rdata_b0_delay),
    .sram_rdata_b1_delay(sram_rdata_b1_delay),
    .sram_rdata_b2_delay(sram_rdata_b2_delay),
    .sram_rdata_b3_delay(sram_rdata_b3_delay),
    .base_0(base[84]),
    .base_1(base[85]),
    .base_2(base[86]),
    .base_3(base[87]),
    .map_0(ch21_0),
    .map_1(ch21_1),
    .map_2(ch21_2),
    .map_3(ch21_3),
    .map_4(ch21_4),
    .map_5(ch21_5),
    .map_6(ch21_6),
    .map_7(ch21_7),
    .map_8(ch21_8),
    .map_9(ch21_9),
    .map_10(ch21_10),
    .map_11(ch21_11),
    .map_12(ch21_12),
    .map_13(ch21_13),
    .map_14(ch21_14),
    .map_15(ch21_15)
);
map_generator map22(
    .map_type_delay(map_type_delay),
    .state(state),
    .sram_rdata_a0_delay(sram_rdata_a0_delay),
    .sram_rdata_a1_delay(sram_rdata_a1_delay),
    .sram_rdata_a2_delay(sram_rdata_a2_delay),
    .sram_rdata_a3_delay(sram_rdata_a3_delay),
    .sram_rdata_b0_delay(sram_rdata_b0_delay),
    .sram_rdata_b1_delay(sram_rdata_b1_delay),
    .sram_rdata_b2_delay(sram_rdata_b2_delay),
    .sram_rdata_b3_delay(sram_rdata_b3_delay),
    .base_0(base[88]),
    .base_1(base[89]),
    .base_2(base[90]),
    .base_3(base[91]),
    .map_0(ch22_0),
    .map_1(ch22_1),
    .map_2(ch22_2),
    .map_3(ch22_3),
    .map_4(ch22_4),
    .map_5(ch22_5),
    .map_6(ch22_6),
    .map_7(ch22_7),
    .map_8(ch22_8),
    .map_9(ch22_9),
    .map_10(ch22_10),
    .map_11(ch22_11),
    .map_12(ch22_12),
    .map_13(ch22_13),
    .map_14(ch22_14),
    .map_15(ch22_15)
);
map_generator map23(
    .map_type_delay(map_type_delay),
    .state(state),
    .sram_rdata_a0_delay(sram_rdata_a0_delay),
    .sram_rdata_a1_delay(sram_rdata_a1_delay),
    .sram_rdata_a2_delay(sram_rdata_a2_delay),
    .sram_rdata_a3_delay(sram_rdata_a3_delay),
    .sram_rdata_b0_delay(sram_rdata_b0_delay),
    .sram_rdata_b1_delay(sram_rdata_b1_delay),
    .sram_rdata_b2_delay(sram_rdata_b2_delay),
    .sram_rdata_b3_delay(sram_rdata_b3_delay),
    .base_0(base[92]),
    .base_1(base[93]),
    .base_2(base[94]),
    .base_3(base[95]),
    .map_0(ch23_0),
    .map_1(ch23_1),
    .map_2(ch23_2),
    .map_3(ch23_3),
    .map_4(ch23_4),
    .map_5(ch23_5),
    .map_6(ch23_6),
    .map_7(ch23_7),
    .map_8(ch23_8),
    .map_9(ch23_9),
    .map_10(ch23_10),
    .map_11(ch23_11),
    .map_12(ch23_12),
    .map_13(ch23_13),
    .map_14(ch23_14),
    .map_15(ch23_15)
);

//conv_mul
conv_mul mul_0(
    .clk(clk),
    .rst_n(rst_n),
    .sram_rdata_weight_delay(sram_rdata_weight_delay),
    .base(base_w[0]),
    .map_0(ch0_0),
    .map_1(ch0_1),
    .map_2(ch0_2),
    .map_3(ch0_3),
    .map_4(ch0_4),
    .map_5(ch0_5),
    .map_6(ch0_6),
    .map_7(ch0_7),
    .map_8(ch0_8),
    .map_9(ch0_9),
    .map_10(ch0_10),
    .map_11(ch0_11),
    .map_12(ch0_12),
    .map_13(ch0_13),
    .map_14(ch0_14),
    .map_15(ch0_15),
    .LU_sum(ch0_LU_sum),
    .RU_sum(ch0_RU_sum),
    .LD_sum(ch0_LD_sum),
    .RD_sum(ch0_RD_sum)
);
conv_mul mul_1(
    .clk(clk),
    .rst_n(rst_n),
    .sram_rdata_weight_delay(sram_rdata_weight_delay),
    .base(base_w[1]),
    .map_0(ch1_0),
    .map_1(ch1_1),
    .map_2(ch1_2),
    .map_3(ch1_3),
    .map_4(ch1_4),
    .map_5(ch1_5),
    .map_6(ch1_6),
    .map_7(ch1_7),
    .map_8(ch1_8),
    .map_9(ch1_9),
    .map_10(ch1_10),
    .map_11(ch1_11),
    .map_12(ch1_12),
    .map_13(ch1_13),
    .map_14(ch1_14),
    .map_15(ch1_15),
    .LU_sum(ch1_LU_sum),
    .RU_sum(ch1_RU_sum),
    .LD_sum(ch1_LD_sum),
    .RD_sum(ch1_RD_sum)
);
conv_mul mul_2(
    .clk(clk),
    .rst_n(rst_n),
    .sram_rdata_weight_delay(sram_rdata_weight_delay),
    .base(base_w[2]),
    .map_0(ch2_0),
    .map_1(ch2_1),
    .map_2(ch2_2),
    .map_3(ch2_3),
    .map_4(ch2_4),
    .map_5(ch2_5),
    .map_6(ch2_6),
    .map_7(ch2_7),
    .map_8(ch2_8),
    .map_9(ch2_9),
    .map_10(ch2_10),
    .map_11(ch2_11),
    .map_12(ch2_12),
    .map_13(ch2_13),
    .map_14(ch2_14),
    .map_15(ch2_15),
    .LU_sum(ch2_LU_sum),
    .RU_sum(ch2_RU_sum),
    .LD_sum(ch2_LD_sum),
    .RD_sum(ch2_RD_sum)
);
conv_mul mul_3(
    .clk(clk),
    .rst_n(rst_n),
    .sram_rdata_weight_delay(sram_rdata_weight_delay),
    .base(base_w[3]),
    .map_0(ch3_0),
    .map_1(ch3_1),
    .map_2(ch3_2),
    .map_3(ch3_3),
    .map_4(ch3_4),
    .map_5(ch3_5),
    .map_6(ch3_6),
    .map_7(ch3_7),
    .map_8(ch3_8),
    .map_9(ch3_9),
    .map_10(ch3_10),
    .map_11(ch3_11),
    .map_12(ch3_12),
    .map_13(ch3_13),
    .map_14(ch3_14),
    .map_15(ch3_15),
    .LU_sum(ch3_LU_sum),
    .RU_sum(ch3_RU_sum),
    .LD_sum(ch3_LD_sum),
    .RD_sum(ch3_RD_sum)
);
conv_mul mul_4(
    .clk(clk),
    .rst_n(rst_n),
    .sram_rdata_weight_delay(sram_rdata_weight_delay),
    .base(base_w[4]),
    .map_0(ch4_0),
    .map_1(ch4_1),
    .map_2(ch4_2),
    .map_3(ch4_3),
    .map_4(ch4_4),
    .map_5(ch4_5),
    .map_6(ch4_6),
    .map_7(ch4_7),
    .map_8(ch4_8),
    .map_9(ch4_9),
    .map_10(ch4_10),
    .map_11(ch4_11),
    .map_12(ch4_12),
    .map_13(ch4_13),
    .map_14(ch4_14),
    .map_15(ch4_15),
    .LU_sum(ch4_LU_sum),
    .RU_sum(ch4_RU_sum),
    .LD_sum(ch4_LD_sum),
    .RD_sum(ch4_RD_sum)
);
conv_mul mul_5(
    .clk(clk),
    .rst_n(rst_n),
    .sram_rdata_weight_delay(sram_rdata_weight_delay),
    .base(base_w[5]),
    .map_0(ch5_0),
    .map_1(ch5_1),
    .map_2(ch5_2),
    .map_3(ch5_3),
    .map_4(ch5_4),
    .map_5(ch5_5),
    .map_6(ch5_6),
    .map_7(ch5_7),
    .map_8(ch5_8),
    .map_9(ch5_9),
    .map_10(ch5_10),
    .map_11(ch5_11),
    .map_12(ch5_12),
    .map_13(ch5_13),
    .map_14(ch5_14),
    .map_15(ch5_15),
    .LU_sum(ch5_LU_sum),
    .RU_sum(ch5_RU_sum),
    .LD_sum(ch5_LD_sum),
    .RD_sum(ch5_RD_sum)
);
conv_mul mul_6(
    .clk(clk),
    .rst_n(rst_n),
    .sram_rdata_weight_delay(sram_rdata_weight_delay),
    .base(base_w[6]),
    .map_0(ch6_0),
    .map_1(ch6_1),
    .map_2(ch6_2),
    .map_3(ch6_3),
    .map_4(ch6_4),
    .map_5(ch6_5),
    .map_6(ch6_6),
    .map_7(ch6_7),
    .map_8(ch6_8),
    .map_9(ch6_9),
    .map_10(ch6_10),
    .map_11(ch6_11),
    .map_12(ch6_12),
    .map_13(ch6_13),
    .map_14(ch6_14),
    .map_15(ch6_15),
    .LU_sum(ch6_LU_sum),
    .RU_sum(ch6_RU_sum),
    .LD_sum(ch6_LD_sum),
    .RD_sum(ch6_RD_sum)
);
conv_mul mul_7(
    .clk(clk),
    .rst_n(rst_n),
    .sram_rdata_weight_delay(sram_rdata_weight_delay),
    .base(base_w[7]),
    .map_0(ch7_0),
    .map_1(ch7_1),
    .map_2(ch7_2),
    .map_3(ch7_3),
    .map_4(ch7_4),
    .map_5(ch7_5),
    .map_6(ch7_6),
    .map_7(ch7_7),
    .map_8(ch7_8),
    .map_9(ch7_9),
    .map_10(ch7_10),
    .map_11(ch7_11),
    .map_12(ch7_12),
    .map_13(ch7_13),
    .map_14(ch7_14),
    .map_15(ch7_15),
    .LU_sum(ch7_LU_sum),
    .RU_sum(ch7_RU_sum),
    .LD_sum(ch7_LD_sum),
    .RD_sum(ch7_RD_sum)
);
conv_mul mul_8(
    .clk(clk),
    .rst_n(rst_n),
    .sram_rdata_weight_delay(sram_rdata_weight_delay),
    .base(base_w[8]),
    .map_0(ch8_0),
    .map_1(ch8_1),
    .map_2(ch8_2),
    .map_3(ch8_3),
    .map_4(ch8_4),
    .map_5(ch8_5),
    .map_6(ch8_6),
    .map_7(ch8_7),
    .map_8(ch8_8),
    .map_9(ch8_9),
    .map_10(ch8_10),
    .map_11(ch8_11),
    .map_12(ch8_12),
    .map_13(ch8_13),
    .map_14(ch8_14),
    .map_15(ch8_15),
    .LU_sum(ch8_LU_sum),
    .RU_sum(ch8_RU_sum),
    .LD_sum(ch8_LD_sum),
    .RD_sum(ch8_RD_sum)
);
conv_mul mul_9(
    .clk(clk),
    .rst_n(rst_n),
    .sram_rdata_weight_delay(sram_rdata_weight_delay),
    .base(base_w[9]),
    .map_0(ch9_0),
    .map_1(ch9_1),
    .map_2(ch9_2),
    .map_3(ch9_3),
    .map_4(ch9_4),
    .map_5(ch9_5),
    .map_6(ch9_6),
    .map_7(ch9_7),
    .map_8(ch9_8),
    .map_9(ch9_9),
    .map_10(ch9_10),
    .map_11(ch9_11),
    .map_12(ch9_12),
    .map_13(ch9_13),
    .map_14(ch9_14),
    .map_15(ch9_15),
    .LU_sum(ch9_LU_sum),
    .RU_sum(ch9_RU_sum),
    .LD_sum(ch9_LD_sum),
    .RD_sum(ch9_RD_sum)
);
conv_mul mul_10(
    .clk(clk),
    .rst_n(rst_n),
    .sram_rdata_weight_delay(sram_rdata_weight_delay),
    .base(base_w[10]),
    .map_0(ch10_0),
    .map_1(ch10_1),
    .map_2(ch10_2),
    .map_3(ch10_3),
    .map_4(ch10_4),
    .map_5(ch10_5),
    .map_6(ch10_6),
    .map_7(ch10_7),
    .map_8(ch10_8),
    .map_9(ch10_9),
    .map_10(ch10_10),
    .map_11(ch10_11),
    .map_12(ch10_12),
    .map_13(ch10_13),
    .map_14(ch10_14),
    .map_15(ch10_15),
    .LU_sum(ch10_LU_sum),
    .RU_sum(ch10_RU_sum),
    .LD_sum(ch10_LD_sum),
    .RD_sum(ch10_RD_sum)
);
conv_mul mul_11(
    .clk(clk),
    .rst_n(rst_n),
    .sram_rdata_weight_delay(sram_rdata_weight_delay),
    .base(base_w[11]),
    .map_0(ch11_0),
    .map_1(ch11_1),
    .map_2(ch11_2),
    .map_3(ch11_3),
    .map_4(ch11_4),
    .map_5(ch11_5),
    .map_6(ch11_6),
    .map_7(ch11_7),
    .map_8(ch11_8),
    .map_9(ch11_9),
    .map_10(ch11_10),
    .map_11(ch11_11),
    .map_12(ch11_12),
    .map_13(ch11_13),
    .map_14(ch11_14),
    .map_15(ch11_15),
    .LU_sum(ch11_LU_sum),
    .RU_sum(ch11_RU_sum),
    .LD_sum(ch11_LD_sum),
    .RD_sum(ch11_RD_sum)
);
conv_mul mul_12(
    .clk(clk),
    .rst_n(rst_n),
    .sram_rdata_weight_delay(sram_rdata_weight_delay),
    .base(base_w[12]),
    .map_0(ch12_0),
    .map_1(ch12_1),
    .map_2(ch12_2),
    .map_3(ch12_3),
    .map_4(ch12_4),
    .map_5(ch12_5),
    .map_6(ch12_6),
    .map_7(ch12_7),
    .map_8(ch12_8),
    .map_9(ch12_9),
    .map_10(ch12_10),
    .map_11(ch12_11),
    .map_12(ch12_12),
    .map_13(ch12_13),
    .map_14(ch12_14),
    .map_15(ch12_15),
    .LU_sum(ch12_LU_sum),
    .RU_sum(ch12_RU_sum),
    .LD_sum(ch12_LD_sum),
    .RD_sum(ch12_RD_sum)
);
conv_mul mul_13(
    .clk(clk),
    .rst_n(rst_n),
    .sram_rdata_weight_delay(sram_rdata_weight_delay),
    .base(base_w[13]),
    .map_0(ch13_0),
    .map_1(ch13_1),
    .map_2(ch13_2),
    .map_3(ch13_3),
    .map_4(ch13_4),
    .map_5(ch13_5),
    .map_6(ch13_6),
    .map_7(ch13_7),
    .map_8(ch13_8),
    .map_9(ch13_9),
    .map_10(ch13_10),
    .map_11(ch13_11),
    .map_12(ch13_12),
    .map_13(ch13_13),
    .map_14(ch13_14),
    .map_15(ch13_15),
    .LU_sum(ch13_LU_sum),
    .RU_sum(ch13_RU_sum),
    .LD_sum(ch13_LD_sum),
    .RD_sum(ch13_RD_sum)
);
conv_mul mul_14(
    .clk(clk),
    .rst_n(rst_n),
    .sram_rdata_weight_delay(sram_rdata_weight_delay),
    .base(base_w[14]),
    .map_0(ch14_0),
    .map_1(ch14_1),
    .map_2(ch14_2),
    .map_3(ch14_3),
    .map_4(ch14_4),
    .map_5(ch14_5),
    .map_6(ch14_6),
    .map_7(ch14_7),
    .map_8(ch14_8),
    .map_9(ch14_9),
    .map_10(ch14_10),
    .map_11(ch14_11),
    .map_12(ch14_12),
    .map_13(ch14_13),
    .map_14(ch14_14),
    .map_15(ch14_15),
    .LU_sum(ch14_LU_sum),
    .RU_sum(ch14_RU_sum),
    .LD_sum(ch14_LD_sum),
    .RD_sum(ch14_RD_sum)
);
conv_mul mul_15(
    .clk(clk),
    .rst_n(rst_n),
    .sram_rdata_weight_delay(sram_rdata_weight_delay),
    .base(base_w[15]),
    .map_0(ch15_0),
    .map_1(ch15_1),
    .map_2(ch15_2),
    .map_3(ch15_3),
    .map_4(ch15_4),
    .map_5(ch15_5),
    .map_6(ch15_6),
    .map_7(ch15_7),
    .map_8(ch15_8),
    .map_9(ch15_9),
    .map_10(ch15_10),
    .map_11(ch15_11),
    .map_12(ch15_12),
    .map_13(ch15_13),
    .map_14(ch15_14),
    .map_15(ch15_15),
    .LU_sum(ch15_LU_sum),
    .RU_sum(ch15_RU_sum),
    .LD_sum(ch15_LD_sum),
    .RD_sum(ch15_RD_sum)
);
conv_mul mul_16(
    .clk(clk),
    .rst_n(rst_n),
    .sram_rdata_weight_delay(sram_rdata_weight_delay),
    .base(base_w[16]),
    .map_0(ch16_0),
    .map_1(ch16_1),
    .map_2(ch16_2),
    .map_3(ch16_3),
    .map_4(ch16_4),
    .map_5(ch16_5),
    .map_6(ch16_6),
    .map_7(ch16_7),
    .map_8(ch16_8),
    .map_9(ch16_9),
    .map_10(ch16_10),
    .map_11(ch16_11),
    .map_12(ch16_12),
    .map_13(ch16_13),
    .map_14(ch16_14),
    .map_15(ch16_15),
    .LU_sum(ch16_LU_sum),
    .RU_sum(ch16_RU_sum),
    .LD_sum(ch16_LD_sum),
    .RD_sum(ch16_RD_sum)
);
conv_mul mul_17(
    .clk(clk),
    .rst_n(rst_n),
    .sram_rdata_weight_delay(sram_rdata_weight_delay),
    .base(base_w[17]),
    .map_0(ch17_0),
    .map_1(ch17_1),
    .map_2(ch17_2),
    .map_3(ch17_3),
    .map_4(ch17_4),
    .map_5(ch17_5),
    .map_6(ch17_6),
    .map_7(ch17_7),
    .map_8(ch17_8),
    .map_9(ch17_9),
    .map_10(ch17_10),
    .map_11(ch17_11),
    .map_12(ch17_12),
    .map_13(ch17_13),
    .map_14(ch17_14),
    .map_15(ch17_15),
    .LU_sum(ch17_LU_sum),
    .RU_sum(ch17_RU_sum),
    .LD_sum(ch17_LD_sum),
    .RD_sum(ch17_RD_sum)
);
conv_mul mul_18(
    .clk(clk),
    .rst_n(rst_n),
    .sram_rdata_weight_delay(sram_rdata_weight_delay),
    .base(base_w[18]),
    .map_0(ch18_0),
    .map_1(ch18_1),
    .map_2(ch18_2),
    .map_3(ch18_3),
    .map_4(ch18_4),
    .map_5(ch18_5),
    .map_6(ch18_6),
    .map_7(ch18_7),
    .map_8(ch18_8),
    .map_9(ch18_9),
    .map_10(ch18_10),
    .map_11(ch18_11),
    .map_12(ch18_12),
    .map_13(ch18_13),
    .map_14(ch18_14),
    .map_15(ch18_15),
    .LU_sum(ch18_LU_sum),
    .RU_sum(ch18_RU_sum),
    .LD_sum(ch18_LD_sum),
    .RD_sum(ch18_RD_sum)
);
conv_mul mul_19(
    .clk(clk),
    .rst_n(rst_n),
    .sram_rdata_weight_delay(sram_rdata_weight_delay),
    .base(base_w[19]),
    .map_0(ch19_0),
    .map_1(ch19_1),
    .map_2(ch19_2),
    .map_3(ch19_3),
    .map_4(ch19_4),
    .map_5(ch19_5),
    .map_6(ch19_6),
    .map_7(ch19_7),
    .map_8(ch19_8),
    .map_9(ch19_9),
    .map_10(ch19_10),
    .map_11(ch19_11),
    .map_12(ch19_12),
    .map_13(ch19_13),
    .map_14(ch19_14),
    .map_15(ch19_15),
    .LU_sum(ch19_LU_sum),
    .RU_sum(ch19_RU_sum),
    .LD_sum(ch19_LD_sum),
    .RD_sum(ch19_RD_sum)
);
conv_mul mul_20(
    .clk(clk),
    .rst_n(rst_n),
    .sram_rdata_weight_delay(sram_rdata_weight_delay),
    .base(base_w[20]),
    .map_0(ch20_0),
    .map_1(ch20_1),
    .map_2(ch20_2),
    .map_3(ch20_3),
    .map_4(ch20_4),
    .map_5(ch20_5),
    .map_6(ch20_6),
    .map_7(ch20_7),
    .map_8(ch20_8),
    .map_9(ch20_9),
    .map_10(ch20_10),
    .map_11(ch20_11),
    .map_12(ch20_12),
    .map_13(ch20_13),
    .map_14(ch20_14),
    .map_15(ch20_15),
    .LU_sum(ch20_LU_sum),
    .RU_sum(ch20_RU_sum),
    .LD_sum(ch20_LD_sum),
    .RD_sum(ch20_RD_sum)
);
conv_mul mul_21(
    .clk(clk),
    .rst_n(rst_n),
    .sram_rdata_weight_delay(sram_rdata_weight_delay),
    .base(base_w[21]),
    .map_0(ch21_0),
    .map_1(ch21_1),
    .map_2(ch21_2),
    .map_3(ch21_3),
    .map_4(ch21_4),
    .map_5(ch21_5),
    .map_6(ch21_6),
    .map_7(ch21_7),
    .map_8(ch21_8),
    .map_9(ch21_9),
    .map_10(ch21_10),
    .map_11(ch21_11),
    .map_12(ch21_12),
    .map_13(ch21_13),
    .map_14(ch21_14),
    .map_15(ch21_15),
    .LU_sum(ch21_LU_sum),
    .RU_sum(ch21_RU_sum),
    .LD_sum(ch21_LD_sum),
    .RD_sum(ch21_RD_sum)
);
conv_mul mul_22(
    .clk(clk),
    .rst_n(rst_n),
    .sram_rdata_weight_delay(sram_rdata_weight_delay),
    .base(base_w[22]),
    .map_0(ch22_0),
    .map_1(ch22_1),
    .map_2(ch22_2),
    .map_3(ch22_3),
    .map_4(ch22_4),
    .map_5(ch22_5),
    .map_6(ch22_6),
    .map_7(ch22_7),
    .map_8(ch22_8),
    .map_9(ch22_9),
    .map_10(ch22_10),
    .map_11(ch22_11),
    .map_12(ch22_12),
    .map_13(ch22_13),
    .map_14(ch22_14),
    .map_15(ch22_15),
    .LU_sum(ch22_LU_sum),
    .RU_sum(ch22_RU_sum),
    .LD_sum(ch22_LD_sum),
    .RD_sum(ch22_RD_sum)
);
conv_mul mul_23(
    .clk(clk),
    .rst_n(rst_n),
    .sram_rdata_weight_delay(sram_rdata_weight_delay),
    .base(base_w[23]),
    .map_0(ch23_0),
    .map_1(ch23_1),
    .map_2(ch23_2),
    .map_3(ch23_3),
    .map_4(ch23_4),
    .map_5(ch23_5),
    .map_6(ch23_6),
    .map_7(ch23_7),
    .map_8(ch23_8),
    .map_9(ch23_9),
    .map_10(ch23_10),
    .map_11(ch23_11),
    .map_12(ch23_12),
    .map_13(ch23_13),
    .map_14(ch23_14),
    .map_15(ch23_15),
    .LU_sum(ch23_LU_sum),
    .RU_sum(ch23_RU_sum),
    .LD_sum(ch23_LD_sum),
    .RD_sum(ch23_RD_sum)
);

//conv_sum
conv_sum sum_LU(
    .clk(clk),
    .rst_n(rst_n),
    .state(state),
    .sram_rdata_bias_delay4(sram_rdata_bias_delay4),
    .ch0(ch0_LU_sum),
    .ch1(ch1_LU_sum),
    .ch2(ch2_LU_sum),
    .ch3(ch3_LU_sum),
    .ch4(ch4_LU_sum),
    .ch5(ch5_LU_sum),
    .ch6(ch6_LU_sum),
    .ch7(ch7_LU_sum),
    .ch8(ch8_LU_sum),
    .ch9(ch9_LU_sum),
    .ch10(ch10_LU_sum),
    .ch11(ch11_LU_sum),
    .ch12(ch12_LU_sum),
    .ch13(ch13_LU_sum),
    .ch14(ch14_LU_sum),
    .ch15(ch15_LU_sum),
    .ch16(ch16_LU_sum),
    .ch17(ch17_LU_sum),
    .ch18(ch18_LU_sum),
    .ch19(ch19_LU_sum),
    .ch20(ch20_LU_sum),
    .ch21(ch21_LU_sum),
    .ch22(ch22_LU_sum),
    .ch23(ch23_LU_sum),
    .forwarding(LU_forwarding),
    .pixel_out(LU_out)
);
conv_sum sum_RU(
    .clk(clk),
    .rst_n(rst_n),
    .state(state),
    .sram_rdata_bias_delay4(sram_rdata_bias_delay4),
    .ch0(ch0_RU_sum),
    .ch1(ch1_RU_sum),
    .ch2(ch2_RU_sum),
    .ch3(ch3_RU_sum),
    .ch4(ch4_RU_sum),
    .ch5(ch5_RU_sum),
    .ch6(ch6_RU_sum),
    .ch7(ch7_RU_sum),
    .ch8(ch8_RU_sum),
    .ch9(ch9_RU_sum),
    .ch10(ch10_RU_sum),
    .ch11(ch11_RU_sum),
    .ch12(ch12_RU_sum),
    .ch13(ch13_RU_sum),
    .ch14(ch14_RU_sum),
    .ch15(ch15_RU_sum),
    .ch16(ch16_RU_sum),
    .ch17(ch17_RU_sum),
    .ch18(ch18_RU_sum),
    .ch19(ch19_RU_sum),
    .ch20(ch20_RU_sum),
    .ch21(ch21_RU_sum),
    .ch22(ch22_RU_sum),
    .ch23(ch23_RU_sum),
    .forwarding(RU_forwarding),
    .pixel_out(RU_out)
);
conv_sum sum_LD(
    .clk(clk),
    .rst_n(rst_n),
    .state(state),
    .sram_rdata_bias_delay4(sram_rdata_bias_delay4),
    .ch0(ch0_LD_sum),
    .ch1(ch1_LD_sum),
    .ch2(ch2_LD_sum),
    .ch3(ch3_LD_sum),
    .ch4(ch4_LD_sum),
    .ch5(ch5_LD_sum),
    .ch6(ch6_LD_sum),
    .ch7(ch7_LD_sum),
    .ch8(ch8_LD_sum),
    .ch9(ch9_LD_sum),
    .ch10(ch10_LD_sum),
    .ch11(ch11_LD_sum),
    .ch12(ch12_LD_sum),
    .ch13(ch13_LD_sum),
    .ch14(ch14_LD_sum),
    .ch15(ch15_LD_sum),
    .ch16(ch16_LD_sum),
    .ch17(ch17_LD_sum),
    .ch18(ch18_LD_sum),
    .ch19(ch19_LD_sum),
    .ch20(ch20_LD_sum),
    .ch21(ch21_LD_sum),
    .ch22(ch22_LD_sum),
    .ch23(ch23_LD_sum),
    .forwarding(LD_forwarding),
    .pixel_out(LD_out)
);
conv_sum sum_RD(
    .clk(clk),
    .rst_n(rst_n),
    .state(state),
    .sram_rdata_bias_delay4(sram_rdata_bias_delay4),
    .ch0(ch0_RD_sum),
    .ch1(ch1_RD_sum),
    .ch2(ch2_RD_sum),
    .ch3(ch3_RD_sum),
    .ch4(ch4_RD_sum),
    .ch5(ch5_RD_sum),
    .ch6(ch6_RD_sum),
    .ch7(ch7_RD_sum),
    .ch8(ch8_RD_sum),
    .ch9(ch9_RD_sum),
    .ch10(ch10_RD_sum),
    .ch11(ch11_RD_sum),
    .ch12(ch12_RD_sum),
    .ch13(ch13_RD_sum),
    .ch14(ch14_RD_sum),
    .ch15(ch15_RD_sum),
    .ch16(ch16_RD_sum),
    .ch17(ch17_RD_sum),
    .ch18(ch18_RD_sum),
    .ch19(ch19_RD_sum),
    .ch20(ch20_RD_sum),
    .ch21(ch21_RD_sum),
    .ch22(ch22_RD_sum),
    .ch23(ch23_RD_sum),
    .forwarding(RD_forwarding),
    .pixel_out(RD_out)
);

//sramB write ctl
sramB_write_ctl sramB_write_ctl(
    .clk(clk),
    .rst_n(rst_n),
    .state(state), 
    .map_type_delay5(map_type_delay5),
    .read_addr0_delay5(read_addr0_delay5),
    .read_addr1_delay5(read_addr1_delay5),
    .read_addr2_delay5(read_addr2_delay5),
    .read_addr3_delay5(read_addr3_delay5),
    .fmap_idx_delay5(fmap_idx_delay5),
    .LU_out(LU_out),
    .RU_out(RU_out),
    .LD_out(LD_out),
    .RD_out(RD_out),
    .output_en(output_en),
    .sram_wen_a0(sram_wen_a0),
    .sram_wen_a1(sram_wen_a1),
    .sram_wen_a2(sram_wen_a2),
    .sram_wen_a3(sram_wen_a3),
    .sram_wen_b0(sram_wen_b0),
    .sram_wen_b1(sram_wen_b1),
    .sram_wen_b2(sram_wen_b2),
    .sram_wen_b3(sram_wen_b3),
    .sram_bytemask_b0(sram_bytemask_b0),
    .sram_bytemask_b1(sram_bytemask_b1),
    .sram_bytemask_b2(sram_bytemask_b2),
    .sram_bytemask_b3(sram_bytemask_b3),
    .sram_waddr_b0(sram_waddr_b0),
    .sram_waddr_b1(sram_waddr_b1),
    .sram_waddr_b2(sram_waddr_b2),
    .sram_waddr_b3(sram_waddr_b3),
    .sram_wdata_b0(sram_wdata_b0),
    .sram_wdata_b1(sram_wdata_b1),
    .sram_wdata_b2(sram_wdata_b2),
    .sram_wdata_b3(sram_wdata_b3),
    .pad_end(pad_end)
);

//resblock_forwarding
resblock_forwarding res_forward(
    .map_type_delay4(map_type_delay4),
    .fmap_idx_delay4(fmap_idx_delay4),
    .sram_rdata_b0_delay(sram_rdata_b0_delay),
    .sram_rdata_b1_delay(sram_rdata_b1_delay),
    .sram_rdata_b2_delay(sram_rdata_b2_delay),
    .sram_rdata_b3_delay(sram_rdata_b3_delay),
    .LU_forwarding(LU_forwarding),
    .RU_forwarding(RU_forwarding),
    .LD_forwarding(LD_forwarding),
    .RD_forwarding(RD_forwarding)
);

//----------------main----------------

//delay input data
always @(posedge clk)  begin
    if(~rst_n)  begin
        sram_rdata_a0_delay <= 0;
        sram_rdata_a1_delay <= 0;
        sram_rdata_a2_delay <= 0;
        sram_rdata_a3_delay <= 0;
        sram_rdata_b0_delay <= 0;
        sram_rdata_b1_delay <= 0;
        sram_rdata_b2_delay <= 0;
        sram_rdata_b3_delay <= 0;
        sram_rdata_weight_delay <= 0;
        sram_rdata_bias_delay <= 0;
        // sram_rdata_bias_delay2 <= 0;
        sram_rdata_bias_delay3 <= 0;
        sram_rdata_bias_delay4 <= 0;
    end
    else  begin
        sram_rdata_a0_delay <= sram_rdata_a0;
        sram_rdata_a1_delay <= sram_rdata_a1;
        sram_rdata_a2_delay <= sram_rdata_a2;
        sram_rdata_a3_delay <= sram_rdata_a3;
        sram_rdata_b0_delay <= sram_rdata_b0;
        sram_rdata_b1_delay <= sram_rdata_b1;
        sram_rdata_b2_delay <= sram_rdata_b2;
        sram_rdata_b3_delay <= sram_rdata_b3;
        sram_rdata_weight_delay <= sram_rdata_weight;
        sram_rdata_bias_delay <= sram_rdata_bias;
        // sram_rdata_bias_delay2 <= sram_rdata_bias_delay;
        sram_rdata_bias_delay3 <= sram_rdata_bias_delay;
        sram_rdata_bias_delay4 <= sram_rdata_bias_delay3;
    end
end

//determine map type
always @*  begin
    if(row[0]==0 && col[0]==0)
        map_type = 0;
    else if(row[0]==0 && col[0]==1)
        map_type = 1;
    else if(row[0]==1 && col[0]==0)
        map_type = 2;
    else
        map_type = 3;
end

always @(posedge clk)  begin
    if(~rst_n)  begin
        map_type_delay <= 0;
        // map_type_delay2 <= 0;
        map_type_delay3 <= 0;
        map_type_delay4 <= 0;
        map_type_delay5 <= 0;
    end
    else  begin
        map_type_delay <= map_type;
        // map_type_delay2 <= map_type_delay;
        map_type_delay3 <= map_type_delay;
        map_type_delay4 <= map_type_delay3;
        map_type_delay5 <= map_type_delay4;
    end
end


endmodule