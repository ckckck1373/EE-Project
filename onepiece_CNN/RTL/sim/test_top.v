`timescale 1ns/1ns
`define CYCLE 10

module test_top;

localparam CH_NUM = 24;
localparam ACT_PER_ADDR = 4;
localparam BW_PER_ACT = 16;
localparam WEIGHT_PER_ADDR = 216; 
localparam BIAS_PER_ADDR = 1;
localparam BW_PER_WEIGHT = 8;
localparam BW_PER_BIAS = 8;
localparam BASE_BW = 11;

//-----------module I/O-----------
reg clk;
reg rst_n;
reg enable;

//read data from SRAM group A
wire [CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1:0] sram_rdata_a0;
wire [CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1:0] sram_rdata_a1;
wire [CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1:0] sram_rdata_a2;
wire [CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1:0] sram_rdata_a3;
//read data from SRAM group B
wire [CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1:0] sram_rdata_b0;
wire [CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1:0] sram_rdata_b1;
wire [CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1:0] sram_rdata_b2;
wire [CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1:0] sram_rdata_b3;
//read data from SRAM weight & bias
wire [WEIGHT_PER_ADDR*BW_PER_WEIGHT-1:0] sram_rdata_weight;
wire signed [BIAS_PER_ADDR*BW_PER_BIAS-1:0] sram_rdata_bias;

//read address from SRAM group A
wire [15:0] sram_raddr_a0;
wire [15:0] sram_raddr_a1;
wire [15:0] sram_raddr_a2;
wire [15:0] sram_raddr_a3;
//read address from SRAM group B
wire [15:0] sram_raddr_b0;
wire [15:0] sram_raddr_b1;
wire [15:0] sram_raddr_b2;
wire [15:0] sram_raddr_b3;
//read address from SRAM weight & bias
wire [8:0] sram_raddr_weight;
wire [8:0] sram_raddr_bias;

//output
wire test_layer_finish;
wire valid;

//write enable for SRAM groups A & B
wire sram_wen_a0;
wire sram_wen_a1;
wire sram_wen_a2;
wire sram_wen_a3;

wire sram_wen_b0;
wire sram_wen_b1;
wire sram_wen_b2;
wire sram_wen_b3;

//bytemask for SRAM groups A & B
wire [CH_NUM*ACT_PER_ADDR-1:0] sram_bytemask_b0;
wire [CH_NUM*ACT_PER_ADDR-1:0] sram_bytemask_b1;
wire [CH_NUM*ACT_PER_ADDR-1:0] sram_bytemask_b2;
wire [CH_NUM*ACT_PER_ADDR-1:0] sram_bytemask_b3;

//write addrress to SRAM groups A & B
wire [15:0] sram_waddr_b0;
wire [15:0] sram_waddr_b1;
wire [15:0] sram_waddr_b2;
wire [15:0] sram_waddr_b3;

//write data to SRAM groups A & B
wire [CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1:0] sram_wdata_b0;
wire [CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1:0] sram_wdata_b1;
wire [CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1:0] sram_wdata_b2;
wire [CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1:0] sram_wdata_b3;

//test_layer
parameter CONV1 = 2, RES1_1 = 3, RES1_2 = 4, RES4_2 = 5, UP_1 = 6, UP_2 = 7, CONV2 = 8;

//layer
integer test_layer;

//verification
reg signed [BW_PER_ACT-1:0] RTL, GOL;
integer offset;
reg wrong;

//-----------top connection-----------
top 
// #(
//     .CH_NUM(CH_NUM),
//     .ACT_PER_ADDR(ACT_PER_ADDR),
//     .BW_PER_ACT(BW_PER_ACT),
//     .WEIGHT_PER_ADDR(WEIGHT_PER_ADDR),
//     .BIAS_PER_ADDR(BIAS_PER_ADDR),
//     .BW_PER_WEIGHT(BW_PER_WEIGHT),
//     .BW_PER_BIAS(BW_PER_BIAS),
//     .BASE_BW(BASE_BW)
// )
top (
    .clk(clk),
    .rst_n(rst_n),
    .enable(enable),

    .sram_rdata_a0(sram_rdata_a0),
    .sram_rdata_a1(sram_rdata_a1),
    .sram_rdata_a2(sram_rdata_a2),
    .sram_rdata_a3(sram_rdata_a3),
    .sram_rdata_b0(sram_rdata_b0),
    .sram_rdata_b1(sram_rdata_b1),
    .sram_rdata_b2(sram_rdata_b2),
    .sram_rdata_b3(sram_rdata_b3),

    .sram_rdata_weight(sram_rdata_weight),
    .sram_rdata_bias(sram_rdata_bias),

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

    .test_layer_finish(test_layer_finish),
    .valid(valid),
    
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
    .sram_wdata_b3(sram_wdata_b3)
);

//-----------sram connection-----------
//wight sram
sram_24x864b sram_24x864b_weight(
    .clk(clk),
    .csb(1'b0),
    .wsb(1'b1),
    .wdata(1728'd0),
    .waddr(9'd0),
    .raddr(sram_raddr_weight),
    .rdata(sram_rdata_weight)
);
//bias sram
sram_24x4b sram_24x4b_bias(
    .clk(clk),
    .csb(1'b0),
    .wsb(1'b1),
    .wdata(8'd0),
    .waddr(9'd0),
    .raddr(sram_raddr_bias),
    .rdata(sram_rdata_bias)
);
//activation sram group A
sram_58101x1056b sram_58101x1056b_a0(
    .clk(clk),
    .bytemask(sram_bytemask_b0),
    .csb(1'b0),
    .wsb(sram_wen_a0),
    .wdata(sram_wdata_b0),
    .waddr(sram_waddr_b0),
    .raddr(sram_raddr_a0),
    .rdata(sram_rdata_a0)
);
sram_58101x1056b sram_58101x1056b_a1(
    .clk(clk),
    .bytemask(sram_bytemask_b1),
    .csb(1'b0),
    .wsb(sram_wen_a1),
    .wdata(sram_wdata_b1),
    .waddr(sram_waddr_b1),
    .raddr(sram_raddr_a1),
    .rdata(sram_rdata_a1)
);
sram_58101x1056b sram_58101x1056b_a2(
    .clk(clk),
    .bytemask(sram_bytemask_b2),
    .csb(1'b0),
    .wsb(sram_wen_a2),
    .wdata(sram_wdata_b2),
    .waddr(sram_waddr_b2),
    .raddr(sram_raddr_a2),
    .rdata(sram_rdata_a2)
);
sram_58101x1056b sram_58101x1056b_a3(
    .clk(clk),
    .bytemask(sram_bytemask_b3),
    .csb(1'b0),
    .wsb(sram_wen_a3),
    .wdata(sram_wdata_b3),
    .waddr(sram_waddr_b3),
    .raddr(sram_raddr_a3),
    .rdata(sram_rdata_a3)
);
//activation sram group B
sram_58101x1056b sram_58101x1056b_b0(
    .clk(clk),
    .bytemask(sram_bytemask_b0),
    .csb(1'b0),
    .wsb(sram_wen_b0),
    .wdata(sram_wdata_b0),
    .waddr(sram_waddr_b0),
    .raddr(sram_raddr_b0),
    .rdata(sram_rdata_b0)
);
sram_58101x1056b sram_58101x1056b_b1(
    .clk(clk),
    .bytemask(sram_bytemask_b1),
    .csb(1'b0),
    .wsb(sram_wen_b1),
    .wdata(sram_wdata_b1),
    .waddr(sram_waddr_b1),
    .raddr(sram_raddr_b1),
    .rdata(sram_rdata_b1)
);
sram_58101x1056b sram_58101x1056b_b2(
    .clk(clk),
    .bytemask(sram_bytemask_b2),
    .csb(1'b0),
    .wsb(sram_wen_b2),
    .wdata(sram_wdata_b2),
    .waddr(sram_waddr_b2),
    .raddr(sram_raddr_b2),
    .rdata(sram_rdata_b2)
);
sram_58101x1056b sram_58101x1056b_b3(
    .clk(clk),
    .bytemask(sram_bytemask_b3),
    .csb(1'b0),
    .wsb(sram_wen_b3),
    .wdata(sram_wdata_b3),
    .waddr(sram_waddr_b3),
    .raddr(sram_raddr_b3),
    .rdata(sram_rdata_b3)
);

//-----------Dump waveform-----------
// initial  begin
//     $fsdbDumpfile("One_sum.fsdb");
// //     // $fsdbDumpvars();
//     // $fsdbDumpvars(0, top.sum_RD);
//     $fsdbDumpvars(0, top.sum_LD);
//     // $fsdbDumpvars(0, top.sum_LU);
//     // $fsdbDumpvars(0, top.sum_RU);
//     $fsdbDumpvars(0, top.sramB_write_ctl);
// //     // #(`CYCLE*100) $finish;
// end
// initial  begin
//     $fsdbDumpfile("One_write.fsdb");
//     // $fsdbDumpvars();
//     $fsdbDumpvars(0, top.sum_RD);
//     $fsdbDumpvars(0, top.sramB_write_ctl);
// //     // #(`CYCLE*100) $finish;
// end


//-----------Store parameters to SRAM-----------
// weight & bias
reg [WEIGHT_PER_ADDR*BW_PER_WEIGHT-1:0] weight [0:410];
reg [BIAS_PER_ADDR*BW_PER_BIAS-1:0] bias [0:410];
// input image
reg [CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1:0] input_image_bank0 [0:3725];
reg [CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1:0] input_image_bank1 [0:3725];
reg [CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1:0] input_image_bank2 [0:3725];
reg [CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1:0] input_image_bank3 [0:3725];
// conv1_golden
reg [CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1:0] conv1_golden_bank0 [0:3725];
reg [CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1:0] conv1_golden_bank1 [0:3725];
reg [CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1:0] conv1_golden_bank2 [0:3725];
reg [CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1:0] conv1_golden_bank3 [0:3725];
// RES1_1_golden
reg [CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1:0] RES1_1_golden_bank0 [0:3725];
reg [CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1:0] RES1_1_golden_bank1 [0:3725];
reg [CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1:0] RES1_1_golden_bank2 [0:3725];
reg [CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1:0] RES1_1_golden_bank3 [0:3725];
// RES1_2_golden
reg [CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1:0] RES1_2_golden_bank0 [0:3725];
reg [CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1:0] RES1_2_golden_bank1 [0:3725];
reg [CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1:0] RES1_2_golden_bank2 [0:3725];
reg [CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1:0] RES1_2_golden_bank3 [0:3725];
// RES4_2_golden
reg [CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1:0] RES4_2_golden_bank0 [0:3725];
reg [CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1:0] RES4_2_golden_bank1 [0:3725];
reg [CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1:0] RES4_2_golden_bank2 [0:3725];
reg [CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1:0] RES4_2_golden_bank3 [0:3725];
// UP_1_golden
reg [CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1:0] UP_1_golden_bank0 [0:14650];
reg [CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1:0] UP_1_golden_bank1 [0:14650];
reg [CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1:0] UP_1_golden_bank2 [0:14650];
reg [CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1:0] UP_1_golden_bank3 [0:14650];
// UP_2_golden
reg [CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1:0] UP_2_golden_bank0 [0:58100];
reg [CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1:0] UP_2_golden_bank1 [0:58100];
reg [CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1:0] UP_2_golden_bank2 [0:58100];
reg [CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1:0] UP_2_golden_bank3 [0:58100];
// conv2_golden
reg [CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1:0] conv2_golden_bank0 [0:58100];
reg [CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1:0] conv2_golden_bank1 [0:58100];
reg [CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1:0] conv2_golden_bank2 [0:58100];
reg [CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1:0] conv2_golden_bank3 [0:58100];

integer i, j;
integer n, mem_idx;
integer pixel = 0;

initial  begin
    //load parameters & input data
    $readmemb("parameters/weight.dat", weight);
    $readmemb("parameters/bias.dat", bias);
    $readmemb("input_image/input_image_bank0.dat", input_image_bank0);
    $readmemb("input_image/input_image_bank1.dat", input_image_bank1);
    $readmemb("input_image/input_image_bank2.dat", input_image_bank2);
    $readmemb("input_image/input_image_bank3.dat", input_image_bank3);

    //laod golden data
    //conv1
    $readmemb("golden/conv1_golden_bank0.dat", conv1_golden_bank0);
    $readmemb("golden/conv1_golden_bank1.dat", conv1_golden_bank1);
    $readmemb("golden/conv1_golden_bank2.dat", conv1_golden_bank2);
    $readmemb("golden/conv1_golden_bank3.dat", conv1_golden_bank3);
    //RES1_1
    $readmemb("golden/RES1_1_golden_bank0.dat", RES1_1_golden_bank0);
    $readmemb("golden/RES1_1_golden_bank1.dat", RES1_1_golden_bank1);
    $readmemb("golden/RES1_1_golden_bank2.dat", RES1_1_golden_bank2);
    $readmemb("golden/RES1_1_golden_bank3.dat", RES1_1_golden_bank3);
    //RES1_2
    $readmemb("golden/RES1_2_forward_golden_bank0.dat", RES1_2_golden_bank0);
    $readmemb("golden/RES1_2_forward_golden_bank1.dat", RES1_2_golden_bank1);
    $readmemb("golden/RES1_2_forward_golden_bank2.dat", RES1_2_golden_bank2);
    $readmemb("golden/RES1_2_forward_golden_bank3.dat", RES1_2_golden_bank3);
    //RES4_2
    $readmemb("golden/RES4_2_forward_golden_bank0.dat", RES4_2_golden_bank0);
    $readmemb("golden/RES4_2_forward_golden_bank1.dat", RES4_2_golden_bank1);
    $readmemb("golden/RES4_2_forward_golden_bank2.dat", RES4_2_golden_bank2);
    $readmemb("golden/RES4_2_forward_golden_bank3.dat", RES4_2_golden_bank3);    
    //UP_1
    $readmemb("golden/UP_1_golden_bank0.dat", UP_1_golden_bank0);
    $readmemb("golden/UP_1_golden_bank1.dat", UP_1_golden_bank1);
    $readmemb("golden/UP_1_golden_bank2.dat", UP_1_golden_bank2);
    $readmemb("golden/UP_1_golden_bank3.dat", UP_1_golden_bank3);
    //UP_2
    $readmemb("golden/UP_2_golden_bank0.dat", UP_2_golden_bank0);
    $readmemb("golden/UP_2_golden_bank1.dat", UP_2_golden_bank1);
    $readmemb("golden/UP_2_golden_bank2.dat", UP_2_golden_bank2);
    $readmemb("golden/UP_2_golden_bank3.dat", UP_2_golden_bank3);
    //conv2
    $readmemb("golden/conv2_golden_bank0.dat", conv2_golden_bank0);
    $readmemb("golden/conv2_golden_bank1.dat", conv2_golden_bank1);
    $readmemb("golden/conv2_golden_bank2.dat", conv2_golden_bank2);
    $readmemb("golden/conv2_golden_bank3.dat", conv2_golden_bank3);

    //store weights into sram
    for(i=0; i<=410; i=i+1)  begin
        sram_24x864b_weight.load_param(i, weight[i]);
    end

    //store biases into sram
    for(i=0; i<=410; i=i+1)  begin
        sram_24x4b_bias.load_param(i, bias[i]);
    end

    //store input image into sram groupA
    for(i=0; i<=3725; i=i+1)  begin
        n = i/81;
        sram_58101x1056b_a0.load_param((i+240*n), input_image_bank0[i]);
    end
    for(i=0; i<=3725; i=i+1)  begin
        n = i/81;
        sram_58101x1056b_a1.load_param((i+240*n), input_image_bank1[i]);
    end
    for(i=0; i<=3725; i=i+1)  begin
        n = i/81;
        sram_58101x1056b_a2.load_param((i+240*n), input_image_bank2[i]);
    end
    for(i=0; i<=3725; i=i+1)  begin
        n = i/81;
        sram_58101x1056b_a3.load_param((i+240*n), input_image_bank3[i]);
    end

end

//-----------system reset-----------
always #(`CYCLE/2) clk = ~clk;

// initial begin
//     @(sram_waddr_b0)begin
//         $display("%d",sram_waddr_b0);
//     end 
// end

//print state
// initial  begin
//     wait(top.fsm.state==2)
//     #(`CYCLE) $display("Checking CONV1");

//     wait(top.fsm.state==3)
//     $display("Checking RES1_1");
//     wait(top.fsm.state==4)
//     $display("Checking RES1_2");  
//     wait(top.fsm.state==3)
//     $display("Checking RES2_1");
//     wait(top.fsm.state==4)
//     $display("Checking RES2_2"); 
//     wait(top.fsm.state==3)
//     $display("Checking RES3_1");
//     wait(top.fsm.state==4)
//     $display("Checking RES3_2"); 
//     wait(top.fsm.state==3)
//     $display("Checking RES4_1");
//     wait(top.fsm.state==4)
//     $display("Checking RES4_2");

//     wait(top.fsm.state==5)
//     $display("Checking UP_1");
//     wait(top.fsm.state==6)
//     $display("Checking UP_2");

// end

initial  begin
    clk = 0;
    rst_n = 1;
    enable = 0;
    @(negedge clk)
        // $display("weight[0]: %b", top.sram_raddr_weight[0]);
        // $display("weight[1]: %b", top.sram_raddr_weight[1]);
        // $display("weight[2]: %b", top.sram_raddr_weight[2]);
        // $display("weight[3]: %b", top.sram_raddr_weight[3]);
        // $display("weight[4]: %b", top.sram_raddr_weight[4]);
        // $display("weight[5]: %b", top.sram_raddr_weight[5]);
        // $display("weight[6]: %b", top.sram_raddr_weight[6]);
        // $display("weight[7]: %b", top.sram_raddr_weight[7]);
        // $display("weight[8]: %b", top.sram_raddr_weight[8]);
        // $display("state: %b", top.state);
    @(negedge clk)
        rst_n = 0;
    @(negedge clk)
    @(negedge clk)
    @(negedge clk)
        // $display("weight[0]: %b", top.sram_raddr_weight[0]);
        // $display("weight[1]: %b", top.sram_raddr_weight[1]);
        // $display("weight[2]: %b", top.sram_raddr_weight[2]);
        // $display("weight[3]: %b", top.sram_raddr_weight[3]);
        // $display("weight[4]: %b", top.sram_raddr_weight[4]);
        // $display("weight[5]: %b", top.sram_raddr_weight[5]);
        // $display("weight[6]: %b", top.sram_raddr_weight[6]);
        // $display("weight[7]: %b", top.sram_raddr_weight[7]);
        // $display("weight[8]: %b", top.sram_raddr_weight[8]);
        // $display("state: %b", top.state);
    @(negedge clk)
        rst_n = 1;
    @(negedge clk)
    @(negedge clk)
        enable = 1;

    wait(top.state==CONV1)
    // check padding
    // SRAM B
        $display("Checking padding in SRAM_B...");
        //row 1
        $display("Checking row 1...");
        for(i=0; i<=320; i=i+1)  begin
            if(sram_58101x1056b_b0.mem[i]!==0)  begin
                $display("You have wrong padding in sram B0 addr %d...", i);
                $finish;
            end
        end
        for(i=0; i<=319; i=i+1)  begin
            if(sram_58101x1056b_b1.mem[i]!==0)  begin
                $display("You have wrong padding in sram B1 addr %d...", i);
                $finish;
            end
        end
        //row 2
        $display("Checking row 2...");
        for(i=14445; i<=14525; i=i+1)  begin
            if(sram_58101x1056b_b0.mem[i]!==0)  begin
                $display("You have wrong padding in sram B0 addr %d...", i);
                $finish;
            end
        end
        for(i=14445; i<=14524; i=i+1)  begin
            if(sram_58101x1056b_b1.mem[i]!==0)  begin
                $display("You have wrong padding in sram B1 addr %d...", i);
                $finish;
            end
        end
        //row 3
        $display("Checking row 3...");
        for(i=57780; i<=58100; i=i+1)  begin
            if(sram_58101x1056b_b0.mem[i]!==0)  begin
                $display("You have wrong padding in sram B0 addr %d...", i);
                $finish;
            end
        end
        for(i=57780; i<=58099; i=i+1)  begin
            if(sram_58101x1056b_b1.mem[i]!==0)  begin
                $display("You have wrong padding in sram B1 addr %d...", i);
                $finish;
            end
        end
        //col 1
        $display("Checking col 1...");
        for(i=0; i<=57780; i=i+321)  begin
            if(sram_58101x1056b_b0.mem[i]!==0)  begin
                $display("You have wrong padding in sram B0 addr %d...", i);
                $finish;
            end
        end
        for(i=0; i<=57459; i=i+321)  begin
            if(sram_58101x1056b_b2.mem[i]!==0)  begin
                $display("You have wrong padding in sram B2 addr %d...", i);
                $finish;
            end
        end
        //col 2
        $display("Checking col 2...");
        for(i=80; i<=14525; i=i+321)  begin
            if(sram_58101x1056b_b0.mem[i]!==0)  begin
                $display("You have wrong padding in sram B0 addr %d...", i);
                $finish;
            end
        end
        for(i=80; i<=14204; i=i+321)  begin
            if(sram_58101x1056b_b2.mem[i]!==0)  begin
                $display("You have wrong padding in sram B2 addr %d...", i);
                $finish;
            end
        end
        //col 3
        $display("Checking col 3...");
        for(i=320; i<=58100; i=i+321)  begin
            if(sram_58101x1056b_b0.mem[i]!==0)  begin
                $display("You have wrong padding in sram B0 addr %d...", i);
                $finish;
            end
        end
        for(i=320; i<=57779; i=i+321)  begin
            if(sram_58101x1056b_b2.mem[i]!==0)  begin
                $display("You have wrong padding in sram B2 addr %d...", i);
                $finish;
            end
        end
    // SRAM A
        $display("Checking padding in SRAM_A...");
        //row 1 bank0
        $display("Checking row 1...");
        for(i=0; i<=160; i=i+1)  begin
            for(j=CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1; j>=0; j=j-4*BW_PER_ACT)
                if(sram_58101x1056b_a0.mem[i][j-:BW_PER_ACT]!==0)  begin
                    $display("You have wrong padding in sram A0 addr %d at channel %d: the %d pixel ", i, 23-j/(4*BW_PER_ACT), 0);
                    $finish;
                end
        end
        for(i=0; i<=160; i=i+1)  begin
            for(j=CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1-BW_PER_ACT; j>=0; j=j-4*BW_PER_ACT)
                if(sram_58101x1056b_a0.mem[i][j-:BW_PER_ACT]!==0)  begin
                    $display("You have wrong padding in sram A0 addr %d at channel %d: the %d pixel ", i, 23-j/(4*BW_PER_ACT), 1);
                    $finish;
                end
        end
        //row 1 bank 1
        for(i=0; i<=159; i=i+1)  begin
            for(j=CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1; j>=0; j=j-4*BW_PER_ACT)
                if(sram_58101x1056b_a1.mem[i][j-:BW_PER_ACT]!==0)  begin
                    $display("You have wrong padding in sram A1 addr %d at channel %d: the %d pixel ", i, 23-j/(4*BW_PER_ACT), 0);
                    $finish;
                end
        end
        for(i=0; i<=159; i=i+1)  begin
            for(j=CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1-BW_PER_ACT; j>=0; j=j-4*BW_PER_ACT)
                if(sram_58101x1056b_a1.mem[i][j-:BW_PER_ACT]!==0)  begin
                    $display("You have wrong padding in sram A1 addr %d at channel %d: the %d pixel ", i, 23-j/(4*BW_PER_ACT), 1);
                    $finish;
                end
        end
        //row 2 bank 0
        $display("Checking row 2...");
        for(i=14445; i<=14525; i=i+1)  begin
            for(j=CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1-2*BW_PER_ACT; j>=0; j=j-4*BW_PER_ACT)
                if(sram_58101x1056b_a0.mem[i][j-:BW_PER_ACT]!==0)  begin
                    $display("You have wrong padding in sram A0 addr %d at channel %d: the %d pixel ", i, 23-j/(4*BW_PER_ACT), 2);
                    $finish;
                end
        end
        for(i=14445; i<=14525; i=i+1)  begin
            for(j=CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1-3*BW_PER_ACT; j>=0; j=j-4*BW_PER_ACT)
                if(sram_58101x1056b_a0.mem[i][j-:BW_PER_ACT]!==0)  begin
                    $display("You have wrong padding in sram A0 addr %d at channel %d: the %d pixel ", i, 23-j/(4*BW_PER_ACT), 3);
                    $finish;
                end
        end
        //row 2 bank 1
        for(i=14445; i<=14524; i=i+1)  begin
            for(j=CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1-2*BW_PER_ACT; j>=0; j=j-4*BW_PER_ACT)
                if(sram_58101x1056b_a1.mem[i][j-:BW_PER_ACT]!==0)  begin
                    $display("You have wrong padding in sram A1 addr %d at channel %d: the %d pixel ", i, 23-j/(4*BW_PER_ACT), 2);
                    $finish;
                end
        end
        for(i=14445; i<=14524; i=i+1)  begin
            for(j=CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1-3*BW_PER_ACT; j>=0; j=j-4*BW_PER_ACT)
                if(sram_58101x1056b_a1.mem[i][j-:BW_PER_ACT]!==0)  begin
                    $display("You have wrong padding in sram A1 addr %d at channel %d: the %d pixel ", i, 23-j/(4*BW_PER_ACT), 3);
                    $finish;
                end
        end
        // row 3 bank 0
        $display("Checking row 3...");
        for(i=28890; i<=29050; i=i+1)  begin
            for(j=CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1-2*BW_PER_ACT; j>=0; j=j-4*BW_PER_ACT)
                if(sram_58101x1056b_a0.mem[i][j-:BW_PER_ACT]!==0)  begin
                    $display("You have wrong padding in sram A0 addr %d at channel %d: the %d pixel ", i, 23-j/(4*BW_PER_ACT), 2);
                    $finish;
                end
        end
        for(i=28890; i<=29050; i=i+1)  begin
            for(j=CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1-3*BW_PER_ACT; j>=0; j=j-4*BW_PER_ACT)
                if(sram_58101x1056b_a0.mem[i][j-:BW_PER_ACT]!==0)  begin
                    $display("You have wrong padding in sram A0 addr %d at channel %d: the %d pixel ", i, 23-j/(4*BW_PER_ACT), 3);
                    $finish;
                end
        end
        // row 3 bank 1
        for(i=28890; i<=29049; i=i+1)  begin
            for(j=CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1-2*BW_PER_ACT; j>=0; j=j-4*BW_PER_ACT)
                if(sram_58101x1056b_a1.mem[i][j-:BW_PER_ACT]!==0)  begin
                    $display("You have wrong padding in sram A1 addr %d at channel %d: the %d pixel ", i, 23-j/(4*BW_PER_ACT), 2);
                    $finish;
                end
        end
        for(i=28890; i<=29049; i=i+1)  begin
            for(j=CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1-3*BW_PER_ACT; j>=0; j=j-4*BW_PER_ACT)
                if(sram_58101x1056b_a1.mem[i][j-:BW_PER_ACT]!==0)  begin
                    $display("You have wrong padding in sram A1 addr %d at channel %d: the %d pixel ", i, 23-j/(4*BW_PER_ACT), 3);
                    $finish;
                end
        end
        // col 1 bank 0
        $display("Checking col 1...");
        for(i=0; i<=28890; i=i+321)  begin
            for(j=CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1; j>=0; j=j-4*BW_PER_ACT)
                if(sram_58101x1056b_a0.mem[i][j-:BW_PER_ACT]!==0)  begin
                    $display("You have wrong padding in sram A0 addr %d at channel %d: the %d pixel ", i, 23-j/(4*BW_PER_ACT), 0);
                    $finish;
                end
        end
        for(i=0; i<=28890; i=i+321)  begin
            for(j=CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1-2*BW_PER_ACT; j>=0; j=j-4*BW_PER_ACT)
                if(sram_58101x1056b_a0.mem[i][j-:BW_PER_ACT]!==0)  begin
                    $display("You have wrong padding in sram A0 addr %d at channel %d: the %d pixel ", i, 23-j/(4*BW_PER_ACT), 2);
                    $finish;
                end
        end
        // col 1 bank 2
        for(i=0; i<=28569; i=i+321)  begin
            for(j=CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1; j>=0; j=j-4*BW_PER_ACT)
                if(sram_58101x1056b_a2.mem[i][j-:BW_PER_ACT]!==0)  begin
                    $display("You have wrong padding in sram A2 addr %d at channel %d: the %d pixel ", i, 23-j/(4*BW_PER_ACT), 0);
                    $finish;
                end
        end
        for(i=0; i<=28569; i=i+321)  begin
            for(j=CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1-2*BW_PER_ACT; j>=0; j=j-4*BW_PER_ACT)
                if(sram_58101x1056b_a2.mem[i][j-:BW_PER_ACT]!==0)  begin
                    $display("You have wrong padding in sram A2 addr %d at channel %d: the %d pixel ", i, 23-j/(4*BW_PER_ACT), 2);
                    $finish;
                end
        end
        // col 2 bank 0
        $display("Checking col 2...");
        for(i=80; i<=14525; i=i+321)  begin
            for(j=CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1-BW_PER_ACT; j>=0; j=j-4*BW_PER_ACT)
                if(sram_58101x1056b_a0.mem[i][j-:BW_PER_ACT]!==0)  begin
                    $display("You have wrong padding in sram A0 addr %d at channel %d: the %d pixel ", i, 23-j/(4*BW_PER_ACT), 1);
                    $finish;
                end
        end
        for(i=80; i<=14525; i=i+321)  begin
            for(j=CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1-3*BW_PER_ACT; j>=0; j=j-4*BW_PER_ACT)
                if(sram_58101x1056b_a0.mem[i][j-:BW_PER_ACT]!==0)  begin
                    $display("You have wrong padding in sram A0 addr %d at channel %d: the %d pixel ", i, 23-j/(4*BW_PER_ACT), 3);
                    $finish;
                end
        end
        // col 2 bank 2
        for(i=80; i<=14204; i=i+321)  begin
            for(j=CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1-BW_PER_ACT; j>=0; j=j-4*BW_PER_ACT)
                if(sram_58101x1056b_a2.mem[i][j-:BW_PER_ACT]!==0)  begin
                    $display("You have wrong padding in sram A2 addr %d at channel %d: the %d pixel ", i, 23-j/(4*BW_PER_ACT), 1);
                    $finish;
                end
        end
        for(i=80; i<=14204; i=i+321)  begin
            for(j=CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1-3*BW_PER_ACT; j>=0; j=j-4*BW_PER_ACT)
                if(sram_58101x1056b_a2.mem[i][j-:BW_PER_ACT]!==0)  begin
                    $display("You have wrong padding in sram A2 addr %d at channel %d: the %d pixel ", i, 23-j/(4*BW_PER_ACT), 3);
                    $finish;
                end
        end
        // col 3 bank 0
        $display("Checking col 3...");
        for(i=160; i<=29050; i=i+321)  begin
            for(j=CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1-BW_PER_ACT; j>=0; j=j-4*BW_PER_ACT)
                if(sram_58101x1056b_a0.mem[i][j-:BW_PER_ACT]!==0)  begin
                    $display("You have wrong padding in sram A0 addr %d at channel %d: the %d pixel ", i, 23-j/(4*BW_PER_ACT), 1);
                    $finish;
                end
        end
        for(i=160; i<=29050; i=i+321)  begin
            for(j=CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1-3*BW_PER_ACT; j>=0; j=j-4*BW_PER_ACT)
                if(sram_58101x1056b_a0.mem[i][j-:BW_PER_ACT]!==0)  begin
                    $display("You have wrong padding in sram A0 addr %d at channel %d: the %d pixel ", i, 23-j/(4*BW_PER_ACT), 3);
                    $finish;
                end
        end
        // col 3 bank 2
        for(i=160; i<=28729; i=i+321)  begin
            for(j=CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1-BW_PER_ACT; j>=0; j=j-4*BW_PER_ACT)
                if(sram_58101x1056b_a2.mem[i][j-:BW_PER_ACT]!==0)  begin
                    $display("You have wrong padding in sram A2 addr %d at channel %d: the %d pixel ", i, 23-j/(4*BW_PER_ACT), 1);
                    $finish;
                end
        end
        for(i=160; i<=28729; i=i+321)  begin
            for(j=CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1-3*BW_PER_ACT; j>=0; j=j-4*BW_PER_ACT)
                if(sram_58101x1056b_a2.mem[i][j-:BW_PER_ACT]!==0)  begin
                    $display("You have wrong padding in sram A2 addr %d at channel %d: the %d pixel ", i, 23-j/(4*BW_PER_ACT), 3);
                    $finish;
                end
        end
        $display("Padding of SRAM_A and SRAM_B are correct :)\n");


    // wait(top.fsm.state==3)
    wait(test_layer_finish)

    //check the first output pixel
        // $display("R");
        //   $write($signed(sram_58101x1056b_a0.mem[0][(CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1               )-:BW_PER_ACT]));
        //   $write($signed(sram_58101x1056b_a0.mem[0][(CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1-BW_PER_ACT    )-:BW_PER_ACT]));
        // $display($signed(sram_58101x1056b_a1.mem[0][(CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1               )-:BW_PER_ACT]));
        //   $write($signed(sram_58101x1056b_a0.mem[0][(CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1-2*BW_PER_ACT  )-:BW_PER_ACT]));
        //   $write($signed(sram_58101x1056b_a0.mem[0][(CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1-3*BW_PER_ACT  )-:BW_PER_ACT]));
        // $display($signed(sram_58101x1056b_a1.mem[0][(CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1-2*BW_PER_ACT  )-:BW_PER_ACT]));
        //   $write($signed(sram_58101x1056b_a2.mem[0][(CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1               )-:BW_PER_ACT]));
        //   $write($signed(sram_58101x1056b_a2.mem[0][(CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1-BW_PER_ACT    )-:BW_PER_ACT]));
        // $display($signed(sram_58101x1056b_a3.mem[0][(CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1               )-:BW_PER_ACT]));
        
        // $display("G");
        //   $write($signed(sram_58101x1056b_a0.mem[0][(CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1             -4*BW_PER_ACT)-:BW_PER_ACT]));
        //   $write($signed(sram_58101x1056b_a0.mem[0][(CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1-BW_PER_ACT  -4*BW_PER_ACT)-:BW_PER_ACT]));
        // $display($signed(sram_58101x1056b_a1.mem[0][(CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1             -4*BW_PER_ACT)-:BW_PER_ACT]));
        //   $write($signed(sram_58101x1056b_a0.mem[0][(CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1-2*BW_PER_ACT-4*BW_PER_ACT)-:BW_PER_ACT]));
        //   $write($signed(sram_58101x1056b_a0.mem[0][(CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1-3*BW_PER_ACT-4*BW_PER_ACT)-:BW_PER_ACT]));
        // $display($signed(sram_58101x1056b_a1.mem[0][(CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1-2*BW_PER_ACT-4*BW_PER_ACT)-:BW_PER_ACT]));
        //   $write($signed(sram_58101x1056b_a2.mem[0][(CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1             -4*BW_PER_ACT)-:BW_PER_ACT]));
        //   $write($signed(sram_58101x1056b_a2.mem[0][(CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1-BW_PER_ACT  -4*BW_PER_ACT)-:BW_PER_ACT]));
        // $display($signed(sram_58101x1056b_a3.mem[0][(CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1             -4*BW_PER_ACT)-:BW_PER_ACT]));
        
        // $display("B");
        //   $write($signed(sram_58101x1056b_a0.mem[0][(CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1             -8*BW_PER_ACT)-:BW_PER_ACT]));
        //   $write($signed(sram_58101x1056b_a0.mem[0][(CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1-BW_PER_ACT  -8*BW_PER_ACT)-:BW_PER_ACT]));
        // $display($signed(sram_58101x1056b_a1.mem[0][(CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1             -8*BW_PER_ACT)-:BW_PER_ACT]));
        //   $write($signed(sram_58101x1056b_a0.mem[0][(CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1-2*BW_PER_ACT-8*BW_PER_ACT)-:BW_PER_ACT]));
        //   $write($signed(sram_58101x1056b_a0.mem[0][(CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1-3*BW_PER_ACT-8*BW_PER_ACT)-:BW_PER_ACT]));
        // $display($signed(sram_58101x1056b_a1.mem[0][(CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1-2*BW_PER_ACT-8*BW_PER_ACT)-:BW_PER_ACT]));
        //   $write($signed(sram_58101x1056b_a2.mem[0][(CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1             -8*BW_PER_ACT)-:BW_PER_ACT]));
        //   $write($signed(sram_58101x1056b_a2.mem[0][(CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1-BW_PER_ACT  -8*BW_PER_ACT)-:BW_PER_ACT]));
        // $display($signed(sram_58101x1056b_a3.mem[0][(CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1             -8*BW_PER_ACT)-:BW_PER_ACT]));

        // n = 9*CH_NUM*BW_PER_WEIGHT-1;
        // for (j=0; j<=2; j=j+1)  begin
        //     $display("Weight%d:", j);
        //     for(i=0; i<=8; i=i+1)  begin
        //         if(i%3==2)
        //             $write("%d\n",$signed(sram_24x864b_weight.mem[0][n-:BW_PER_WEIGHT]));
        //         else  begin
        //             $write("%d",$signed(sram_24x864b_weight.mem[0][n-:BW_PER_WEIGHT]));
        //             $write(" ");
        //         end
        //         n = n-8;
        //     end
        // end

        // $display("Bias:%d", $signed(sram_24x4b_bias.mem[0]));

        // $display("RTL ans: %b", sram_58101x1056b_b0.mem[0][(CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1-3*BW_PER_ACT)-:BW_PER_ACT]);
        // $display("Gol ans: %b", conv1_golden_bank0[0][(CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1-3*BW_PER_ACT)-:BW_PER_ACT]);

    test_layer = CONV2; // MODIFY
    // If compare CONV1: "CONV1"
    // If compare RES1_1: "RES1_1"
    // If compare RES1_2: "RES1_2"
    // If compare UP_1: "UP_1"
    case(test_layer)
        CONV1:  begin
            $display("Checking Conv1...");
            // $display("%b", sram_58101x1056b_b0.mem[0][(CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1-3*BW_PER_ACT)-:BW_PER_ACT]);
            pixel = 0;
            for(i=0; i<=3725; i=i+1)  begin
                n = i/81;
                mem_idx = i+240*n;
                for(j=CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1; j>0; j=j-BW_PER_ACT)  begin
                    if(pixel==4)  begin
                        pixel = 0;
                    end
                    if(sram_58101x1056b_b0.mem[mem_idx][j-:BW_PER_ACT]!==conv1_golden_bank0[i][j-:BW_PER_ACT])  begin
                        $display("You have wrong answer in sram B0 addr %d...", mem_idx);
                        $display("The %d pixel at channel %d: ",pixel ,23-j/(4*BW_PER_ACT));
                        $display("RTL: %b", sram_58101x1056b_b0.mem[mem_idx][j-:BW_PER_ACT]);
                        $display("GOL: %b", conv1_golden_bank0[i][j-:BW_PER_ACT]);
                        $finish; // MODIFY MULTITEST
                    end
                    pixel = pixel + 1;
                end
            end

            pixel = 0;
            for(i=0; i<=3725; i=i+1)  begin
                n = i/81;
                mem_idx = i+240*n;
                for(j=CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1; j>0; j=j-BW_PER_ACT)  begin
                    if(pixel==4)  begin
                        pixel = 0;
                    end
                    if(sram_58101x1056b_b1.mem[mem_idx][j-:BW_PER_ACT]!==conv1_golden_bank1[i][j-:BW_PER_ACT])  begin
                        $display("You have wrong answer in sram B1 addr %d...", mem_idx);
                        $display("The %d pixel at channel %d: ",pixel ,23-j/(4*BW_PER_ACT));
                        $display("RTL: %b", sram_58101x1056b_b1.mem[mem_idx][j-:BW_PER_ACT]);
                        $display("GOL: %b", conv1_golden_bank1[i][j-:BW_PER_ACT]);
                        $finish;
                    end
                    pixel = pixel + 1;
                end
            end

            pixel = 0;
            for(i=0; i<=3725; i=i+1)  begin
                n = i/81;
                mem_idx = i+240*n;
                for(j=CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1; j>0; j=j-BW_PER_ACT)  begin
                    if(pixel==4)  begin
                        pixel = 0;
                    end
                    if(sram_58101x1056b_b2.mem[mem_idx][j-:BW_PER_ACT]!==conv1_golden_bank2[i][j-:BW_PER_ACT])  begin
                        $display("You have wrong answer in sram B2 addr %d...", mem_idx);
                        $display("The %d pixel at channel %d: ",pixel ,23-j/(4*BW_PER_ACT));
                        $display("RTL: %b", sram_58101x1056b_b2.mem[mem_idx][j-:BW_PER_ACT]);
                        $display("GOL: %b", conv1_golden_bank2[i][j-:BW_PER_ACT]);
                        $finish;
                    end
                    pixel = pixel + 1;
                end
            end

            pixel = 0;
            for(i=0; i<=3725; i=i+1)  begin
                n = i/81;
                mem_idx = i+240*n;
                for(j=CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1; j>0; j=j-BW_PER_ACT)  begin
                    if(pixel==4)  begin
                        pixel = 0;
                    end
                    if(sram_58101x1056b_b3.mem[mem_idx][j-:BW_PER_ACT]!==conv1_golden_bank3[i][j-:BW_PER_ACT])  begin
                        $display("You have wrong answer in sram B3 addr %d...", mem_idx);
                        $display("The %d pixel at channel %d: ",pixel ,23-j/(4*BW_PER_ACT));
                        $display("RTL: %b", sram_58101x1056b_b3.mem[mem_idx][j-:BW_PER_ACT]);
                        $display("GOL: %b", conv1_golden_bank3[i][j-:BW_PER_ACT]);
                        $finish;
                    end
                    pixel = pixel + 1;
                end
            end
            $display("Conv1 is correct :)\n");
        end

        RES1_1:  begin
            $display("Checking RES1_1...");
            pixel = 0;
            for(i=0; i<=3725; i=i+1)  begin
                n = i/81;
                mem_idx = i+240*n;
                for(j=CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1; j>0; j=j-BW_PER_ACT)  begin
                    if(pixel==4)  begin
                        pixel = 0;
                    end
                    if(sram_58101x1056b_a0.mem[mem_idx][j-:BW_PER_ACT]!==RES1_1_golden_bank0[i][j-:BW_PER_ACT])  begin
                        $display("You have wrong answer in sram A0 addr %d...", mem_idx);
                        $display("The %d pixel at channel %d: ",pixel ,23-j/(4*BW_PER_ACT));
                        $display("RTL: %b", sram_58101x1056b_a0.mem[mem_idx][j-:BW_PER_ACT]);
                        $display("GOL: %b", RES1_1_golden_bank0[i][j-:BW_PER_ACT]);
                        //$finish;
                    end
                    pixel = pixel + 1;
                end
            end

            pixel = 0;
            for(i=0; i<=3725; i=i+1)  begin
                n = i/81;
                mem_idx = i+240*n;
                for(j=CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1; j>0; j=j-BW_PER_ACT)  begin
                    if(pixel==4)  begin
                        pixel = 0;
                    end
                    if(sram_58101x1056b_a1.mem[mem_idx][j-:BW_PER_ACT]!==RES1_1_golden_bank1[i][j-:BW_PER_ACT])  begin
                        $display("You have wrong answer in sram A1 addr %d...", mem_idx);
                        $display("The %d pixel at channel %d: ",pixel ,23-j/(4*BW_PER_ACT));
                        $display("RTL: %b", sram_58101x1056b_a1.mem[mem_idx][j-:BW_PER_ACT]);
                        $display("GOL: %b", RES1_1_golden_bank1[i][j-:BW_PER_ACT]);
                        //$finish;
                    end
                    pixel = pixel + 1;
                end
            end

            pixel = 0;
            for(i=0; i<=3725; i=i+1)  begin
                n = i/81;
                mem_idx = i+240*n;
                for(j=CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1; j>0; j=j-BW_PER_ACT)  begin
                    if(pixel==4)  begin
                        pixel = 0;
                    end
                    if(sram_58101x1056b_a2.mem[mem_idx][j-:BW_PER_ACT]!==RES1_1_golden_bank2[i][j-:BW_PER_ACT])  begin
                        $display("You have wrong answer in sram A2 addr %d...", mem_idx);
                        $display("The %d pixel at channel %d: ",pixel ,23-j/(4*BW_PER_ACT));
                        $display("RTL: %b", sram_58101x1056b_a2.mem[mem_idx][j-:BW_PER_ACT]);
                        $display("GOL: %b", RES1_1_golden_bank2[i][j-:BW_PER_ACT]);
                        //$finish;
                    end
                    pixel = pixel + 1;
                end
            end

            pixel = 0;
            for(i=0; i<=3725; i=i+1)  begin
                n = i/81;
                mem_idx = i+240*n;
                for(j=CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1; j>0; j=j-BW_PER_ACT)  begin
                    if(pixel==4)  begin
                        pixel = 0;
                    end
                    if(sram_58101x1056b_a3.mem[mem_idx][j-:BW_PER_ACT]!==RES1_1_golden_bank3[i][j-:BW_PER_ACT])  begin
                        $display("You have wrong answer in sram A3 addr %d...", mem_idx);
                        $display("The %d pixel at channel %d: ",pixel ,23-j/(4*BW_PER_ACT));
                        $display("RTL: %b", sram_58101x1056b_a3.mem[mem_idx][j-:BW_PER_ACT]);
                        $display("GOL: %b", RES1_1_golden_bank3[i][j-:BW_PER_ACT]);
                        //$finish;
                    end
                    pixel = pixel + 1;
                end
            end
            $display("1_RES_1 is correct :)\n");
        end

        RES1_2:  begin
            $display("Checking RES1_2...");
            pixel = 0;
            for(i=0; i<=3725; i=i+1)  begin
                n = i/81;
                mem_idx = i+240*n;
                for(j=CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1; j>0; j=j-BW_PER_ACT)  begin
                    if(pixel==4)  begin
                        pixel = 0;
                    end

                    wrong = 1;
                    for(offset=0; offset<1; offset=offset+1)  begin
                        if(($signed(sram_58101x1056b_b0.mem[mem_idx][j-:BW_PER_ACT])===$signed(RES1_2_golden_bank0[i][j-:BW_PER_ACT])-offset) ||
                           ($signed(sram_58101x1056b_b0.mem[mem_idx][j-:BW_PER_ACT])===$signed(RES1_2_golden_bank0[i][j-:BW_PER_ACT])+offset)    )  begin
                            wrong = 0;
                        end
                    end
                    if(wrong==1)  begin
                        $display("You have wrong answer in sram B0 addr %d...", mem_idx);
                        $display("The %d pixel at channel %d: ",pixel ,23-j/(4*BW_PER_ACT));
                        $display("RTL: %b", sram_58101x1056b_b0.mem[mem_idx][j-:BW_PER_ACT]);
                        $display("GOL: %b", RES1_2_golden_bank0[i][j-:BW_PER_ACT]);
                        $finish;
                    end
                    pixel = pixel + 1;
                end
            end

            pixel = 0;
            for(i=0; i<=3725; i=i+1)  begin
                n = i/81;
                mem_idx = i+240*n;
                for(j=CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1; j>0; j=j-BW_PER_ACT)  begin
                    if(pixel==4)  begin
                        pixel = 0;
                    end

                    wrong = 1;
                    for(offset=0; offset<1; offset=offset+1)  begin
                        if(($signed(sram_58101x1056b_b1.mem[mem_idx][j-:BW_PER_ACT])===$signed(RES1_2_golden_bank1[i][j-:BW_PER_ACT])-offset) ||
                           ($signed(sram_58101x1056b_b1.mem[mem_idx][j-:BW_PER_ACT])===$signed(RES1_2_golden_bank1[i][j-:BW_PER_ACT])+offset)    )  begin
                            wrong = 0;
                        end
                    end
                    if(wrong==1)  begin
                        $display("You have wrong answer in sram B1 addr %d...", mem_idx);
                        $display("The %d pixel at channel %d: ",pixel ,23-j/(4*BW_PER_ACT));
                        $display("RTL: %b", sram_58101x1056b_b1.mem[mem_idx][j-:BW_PER_ACT]);
                        $display("GOL: %b", RES1_2_golden_bank1[i][j-:BW_PER_ACT]);
                        $finish;
                    end
                    pixel = pixel + 1;
                end
            end

            pixel = 0;
            for(i=0; i<=3725; i=i+1)  begin
                n = i/81;
                mem_idx = i+240*n;
                for(j=CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1; j>0; j=j-BW_PER_ACT)  begin
                    if(pixel==4)  begin
                        pixel = 0;
                    end

                    wrong = 1;
                    for(offset=0; offset<1; offset=offset+1)  begin
                        if(($signed(sram_58101x1056b_b2.mem[mem_idx][j-:BW_PER_ACT])===$signed(RES1_2_golden_bank2[i][j-:BW_PER_ACT])-offset) ||
                           ($signed(sram_58101x1056b_b2.mem[mem_idx][j-:BW_PER_ACT])===$signed(RES1_2_golden_bank2[i][j-:BW_PER_ACT])+offset)    )  begin
                            wrong = 0;
                        end
                    end
                    if(wrong==1)  begin
                        $display("You have wrong answer in sram B2 addr %d...", mem_idx);
                        $display("The %d pixel at channel %d: ",pixel ,23-j/(4*BW_PER_ACT));
                        $display("RTL: %b", sram_58101x1056b_b2.mem[mem_idx][j-:BW_PER_ACT]);
                        $display("GOL: %b", RES1_2_golden_bank2[i][j-:BW_PER_ACT]);
                        $finish;
                    end
                    pixel = pixel + 1;
                end
            end

            pixel = 0;
            for(i=0; i<=3725; i=i+1)  begin
                n = i/81;
                mem_idx = i+240*n;
                for(j=CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1; j>0; j=j-BW_PER_ACT)  begin
                    if(pixel==4)  begin
                        pixel = 0;
                    end

                    wrong = 1;
                    for(offset=0; offset<1; offset=offset+1)  begin
                        if(($signed(sram_58101x1056b_b3.mem[mem_idx][j-:BW_PER_ACT])===$signed(RES1_2_golden_bank3[i][j-:BW_PER_ACT])-offset) ||
                           ($signed(sram_58101x1056b_b3.mem[mem_idx][j-:BW_PER_ACT])===$signed(RES1_2_golden_bank3[i][j-:BW_PER_ACT])+offset)    )  begin
                            wrong = 0;
                        end
                    end
                    if(wrong==1)  begin
                        $display("You have wrong answer in sram B3 addr %d...", mem_idx);
                        $display("The %d pixel at channel %d: ",pixel ,23-j/(4*BW_PER_ACT));
                        $display("RTL: %b", sram_58101x1056b_b3.mem[mem_idx][j-:BW_PER_ACT]);
                        $display("GOL: %b", RES1_2_golden_bank3[i][j-:BW_PER_ACT]);
                        $finish;
                    end
                    pixel = pixel + 1;
                end
            end
            $display("1_RES_2 is correct :)\n");
        end

        RES4_2:  begin
        $display("Checking RES4_2...");
            pixel = 0;
            for(i=0; i<=3725; i=i+1)  begin
                n = i/81;
                mem_idx = i+240*n;
                for(j=CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1; j>0; j=j-BW_PER_ACT)  begin
                    if(pixel==4)  begin
                        pixel = 0;
                    end

                    wrong = 1;
                    for(offset=0; offset<1; offset=offset+1)  begin
                        if(($signed(sram_58101x1056b_b0.mem[mem_idx][j-:BW_PER_ACT])===$signed(RES4_2_golden_bank0[i][j-:BW_PER_ACT])-offset) ||
                           ($signed(sram_58101x1056b_b0.mem[mem_idx][j-:BW_PER_ACT])===$signed(RES4_2_golden_bank0[i][j-:BW_PER_ACT])+offset)    )  begin
                            wrong = 0;
                        end
                    end
                    if(wrong==1)  begin
                        $display("You have wrong answer in sram B0 addr %d...", mem_idx);
                        $display("The %d pixel at channel %d: ",pixel ,23-j/(4*BW_PER_ACT));
                        $display("RTL: %b", sram_58101x1056b_b0.mem[mem_idx][j-:BW_PER_ACT]);
                        $display("GOL: %b", RES4_2_golden_bank0[i][j-:BW_PER_ACT]);
                        $finish;
                    end
                    pixel = pixel + 1;
                end
            end

            pixel = 0;
            for(i=0; i<=3725; i=i+1)  begin
                n = i/81;
                mem_idx = i+240*n;
                for(j=CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1; j>0; j=j-BW_PER_ACT)  begin
                    if(pixel==4)  begin
                        pixel = 0;
                    end

                    wrong = 1;
                    for(offset=0; offset<1; offset=offset+1)  begin
                        if(($signed(sram_58101x1056b_b1.mem[mem_idx][j-:BW_PER_ACT])===$signed(RES4_2_golden_bank1[i][j-:BW_PER_ACT])-offset) ||
                           ($signed(sram_58101x1056b_b1.mem[mem_idx][j-:BW_PER_ACT])===$signed(RES4_2_golden_bank1[i][j-:BW_PER_ACT])+offset)    )  begin
                            wrong = 0;
                        end
                    end
                    if(wrong==1)  begin
                        $display("You have wrong answer in sram B1 addr %d...", mem_idx);
                        $display("The %d pixel at channel %d: ",pixel ,23-j/(4*BW_PER_ACT));
                        $display("RTL: %b", sram_58101x1056b_b1.mem[mem_idx][j-:BW_PER_ACT]);
                        $display("GOL: %b", RES4_2_golden_bank1[i][j-:BW_PER_ACT]);
                        $finish;
                    end
                    pixel = pixel + 1;
                end
            end
            
            pixel = 0;
            for(i=0; i<=3725; i=i+1)  begin
                n = i/81;
                mem_idx = i+240*n;
                for(j=CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1; j>0; j=j-BW_PER_ACT)  begin
                    if(pixel==4)  begin
                        pixel = 0;
                    end

                    wrong = 1;
                    for(offset=0; offset<1; offset=offset+1)  begin
                        if(($signed(sram_58101x1056b_b2.mem[mem_idx][j-:BW_PER_ACT])===$signed(RES4_2_golden_bank2[i][j-:BW_PER_ACT])-offset) ||
                           ($signed(sram_58101x1056b_b2.mem[mem_idx][j-:BW_PER_ACT])===$signed(RES4_2_golden_bank2[i][j-:BW_PER_ACT])+offset)    )  begin
                            wrong = 0;
                        end
                    end
                    if(wrong==1)  begin
                        $display("You have wrong answer in sram B2 addr %d...", mem_idx);
                        $display("The %d pixel at channel %d: ",pixel ,23-j/(4*BW_PER_ACT));
                        $display("RTL: %b", sram_58101x1056b_b2.mem[mem_idx][j-:BW_PER_ACT]);
                        $display("GOL: %b", RES4_2_golden_bank2[i][j-:BW_PER_ACT]);
                        $finish;
                    end
                    pixel = pixel + 1;
                end
            end

            pixel = 0;
            for(i=0; i<=3725; i=i+1)  begin
                n = i/81;
                mem_idx = i+240*n;
                for(j=CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1; j>0; j=j-BW_PER_ACT)  begin
                    if(pixel==4)  begin
                        pixel = 0;
                    end

                    wrong = 1;
                    for(offset=0; offset<1; offset=offset+1)  begin
                        if(($signed(sram_58101x1056b_b3.mem[mem_idx][j-:BW_PER_ACT])===$signed(RES4_2_golden_bank3[i][j-:BW_PER_ACT])-offset) ||
                           ($signed(sram_58101x1056b_b3.mem[mem_idx][j-:BW_PER_ACT])===$signed(RES4_2_golden_bank3[i][j-:BW_PER_ACT])+offset)    )  begin
                            wrong = 0;
                        end
                    end
                    if(wrong==1)  begin
                        $display("You have wrong answer in sram B3 addr %d...", mem_idx);
                        $display("The %d pixel at channel %d: ",pixel ,23-j/(4*BW_PER_ACT));
                        $display("RTL: %b", sram_58101x1056b_b3.mem[mem_idx][j-:BW_PER_ACT]);
                        $display("GOL: %b", RES4_2_golden_bank3[i][j-:BW_PER_ACT]);
                        $finish;
                    end
                    pixel = pixel + 1;
                end
            end
            $display("4_RES_2 is correct :)\n");
        end

        UP_1:  begin
        $display("Checking UP_1...");
            pixel = 0;
            for(i=0; i<=14650; i=i+1)  begin
                n = i/161;
                mem_idx = i+160*n;
                for(j=CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1; j>0; j=j-BW_PER_ACT)  begin
                    if(pixel==4)  begin
                        pixel = 0;
                    end
                    // $display("%b", sram_58101x1056b_a0.mem[mem_idx][j-:BW_PER_ACT]);
                    wrong = 1;
                    for(offset=0; offset<1; offset=offset+1)  begin
                        if(($signed(sram_58101x1056b_a0.mem[mem_idx][j-:BW_PER_ACT])===$signed(UP_1_golden_bank0[i][j-:BW_PER_ACT])-offset) ||
                           ($signed(sram_58101x1056b_a0.mem[mem_idx][j-:BW_PER_ACT])===$signed(UP_1_golden_bank0[i][j-:BW_PER_ACT])+offset)    )  begin
                            wrong = 0;
                            
                        end
                    end
                    if(wrong==1)  begin
                        $display("You have wrong answer in sram A0 addr %d...", mem_idx);
                        $display("The %d pixel at channel %d: ",pixel ,23-j/(4*BW_PER_ACT));
                        $display("RTL: %b", sram_58101x1056b_a0.mem[mem_idx][j-:BW_PER_ACT]);
                        $display("GOL: %b", UP_1_golden_bank0[i][j-:BW_PER_ACT]);
                        $finish;
                    end
                    pixel = pixel + 1;
                end
            end

            pixel = 0;
            for(i=0; i<=14650; i=i+1)  begin
                n = i/161;
                mem_idx = i+160*n;
                for(j=CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1; j>0; j=j-BW_PER_ACT)  begin
                    if(pixel==4)  begin
                        pixel = 0;
                    end
                    
                    wrong = 1;
                    for(offset=0; offset<2; offset=offset+1)  begin
                        if(($signed(sram_58101x1056b_a1.mem[mem_idx][j-:BW_PER_ACT])===$signed(UP_1_golden_bank1[i][j-:BW_PER_ACT])-offset) ||
                           ($signed(sram_58101x1056b_a1.mem[mem_idx][j-:BW_PER_ACT])===$signed(UP_1_golden_bank1[i][j-:BW_PER_ACT])+offset)    )  begin
                            wrong = 0;
                            
                        end
                    end
                    if(wrong==1)  begin
                        $display("You have wrong answer in sram A1 addr %d...", mem_idx);
                        $display("The %d pixel at channel %d: ",pixel ,23-j/(4*BW_PER_ACT));
                        $display("RTL: %b", sram_58101x1056b_a1.mem[mem_idx][j-:BW_PER_ACT]);
                        $display("GOL: %b", UP_1_golden_bank1[i][j-:BW_PER_ACT]);
                        $finish;
                    end
                    pixel = pixel + 1;
                end
            end
            
            pixel = 0;
            for(i=0; i<=14650; i=i+1)  begin
                n = i/161;
                mem_idx = i+160*n;
                for(j=CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1; j>0; j=j-BW_PER_ACT)  begin
                    if(pixel==4)  begin
                        pixel = 0;
                    end
                    
                    wrong = 1;
                    for(offset=0; offset<2; offset=offset+1)  begin
                        if(($signed(sram_58101x1056b_a2.mem[mem_idx][j-:BW_PER_ACT])===$signed(UP_1_golden_bank2[i][j-:BW_PER_ACT])-offset) ||
                           ($signed(sram_58101x1056b_a2.mem[mem_idx][j-:BW_PER_ACT])===$signed(UP_1_golden_bank2[i][j-:BW_PER_ACT])+offset)    )  begin
                            wrong = 0;
                            
                        end
                    end
                    if(wrong==1)  begin
                        $display("You have wrong answer in sram A2 addr %d...", mem_idx);
                        $display("The %d pixel at channel %d: ",pixel ,23-j/(4*BW_PER_ACT));
                        $display("RTL: %b", sram_58101x1056b_a2.mem[mem_idx][j-:BW_PER_ACT]);
                        $display("GOL: %b", UP_1_golden_bank2[i][j-:BW_PER_ACT]);
                        $finish;
                    end
                    pixel = pixel + 1;
                end
            end

            pixel = 0;
            for(i=0; i<=14650; i=i+1)  begin
                n = i/161;
                mem_idx = i+160*n;
                for(j=CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1; j>0; j=j-BW_PER_ACT)  begin
                    if(pixel==4)  begin
                        pixel = 0;
                    end
                    
                    wrong = 1;
                    for(offset=0; offset<2; offset=offset+1)  begin
                        if(($signed(sram_58101x1056b_a3.mem[mem_idx][j-:BW_PER_ACT])===$signed(UP_1_golden_bank3[i][j-:BW_PER_ACT])-offset) ||
                           ($signed(sram_58101x1056b_a3.mem[mem_idx][j-:BW_PER_ACT])===$signed(UP_1_golden_bank3[i][j-:BW_PER_ACT])+offset)    )  begin
                            wrong = 0;
                
                        end
                    end
                    if(wrong==1)  begin
                        $display("You have wrong answer in sram A3 addr %d...", mem_idx);
                        $display("The %d pixel at channel %d: ",pixel ,23-j/(4*BW_PER_ACT));
                        $display("RTL: %b", sram_58101x1056b_a3.mem[mem_idx][j-:BW_PER_ACT]);
                        $display("GOL: %b", UP_1_golden_bank3[i][j-:BW_PER_ACT]);
                        $finish;
                    end
                    pixel = pixel + 1;
                end
            end
            $display("UP_1 is correct :)\n");
        end

        UP_2:  begin
        $display("Checking UP_2...");
            pixel = 0;
            for(i=0; i<=58100; i=i+1)  begin
                mem_idx = i;
                for(j=CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1; j>0; j=j-BW_PER_ACT)  begin
                    if(pixel==4)  begin
                        pixel = 0;
                    end

                    wrong = 1;
                    for(offset=0; offset<2; offset=offset+1)  begin
                        if(($signed(sram_58101x1056b_b0.mem[mem_idx][j-:BW_PER_ACT])===$signed(UP_2_golden_bank0[i][j-:BW_PER_ACT])-offset) ||
                           ($signed(sram_58101x1056b_b0.mem[mem_idx][j-:BW_PER_ACT])===$signed(UP_2_golden_bank0[i][j-:BW_PER_ACT])+offset)    )  begin
                            wrong = 0;
                        end
                    end
                    if(wrong==1)  begin
                        $display("You have wrong answer in sram B0 addr %d...", mem_idx);
                        $display("The %d pixel at channel %d: ",pixel ,23-j/(4*BW_PER_ACT));
                        $display("RTL: %b", sram_58101x1056b_b0.mem[mem_idx][j-:BW_PER_ACT]);
                        $display("GOL: %b", UP_2_golden_bank0[i][j-:BW_PER_ACT]);
                        $finish;
                    end
                    pixel = pixel + 1;
                end
            end

            pixel = 0;
            for(i=0; i<=58100; i=i+1)  begin
                mem_idx = i;
                for(j=CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1; j>0; j=j-BW_PER_ACT)  begin
                    if(pixel==4)  begin
                        pixel = 0;
                    end

                    wrong = 1;
                    for(offset=0; offset<2; offset=offset+1)  begin
                        if(($signed(sram_58101x1056b_b1.mem[mem_idx][j-:BW_PER_ACT])===$signed(UP_2_golden_bank1[i][j-:BW_PER_ACT])-offset) ||
                           ($signed(sram_58101x1056b_b1.mem[mem_idx][j-:BW_PER_ACT])===$signed(UP_2_golden_bank1[i][j-:BW_PER_ACT])+offset)    )  begin
                            wrong = 0;
                        end
                    end
                    if(wrong==1)  begin
                        $display("You have wrong answer in sram B1 addr %d...", mem_idx);
                        $display("The %d pixel at channel %d: ",pixel ,23-j/(4*BW_PER_ACT));
                        $display("RTL: %b", sram_58101x1056b_b1.mem[mem_idx][j-:BW_PER_ACT]);
                        $display("GOL: %b", UP_2_golden_bank1[i][j-:BW_PER_ACT]);
                        $finish;
                    end
                    pixel = pixel + 1;
                end
            end

            pixel = 0;
            for(i=0; i<=58100; i=i+1)  begin
                mem_idx = i;
                for(j=CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1; j>0; j=j-BW_PER_ACT)  begin
                    if(pixel==4)  begin
                        pixel = 0;
                    end

                    wrong = 1;
                    for(offset=0; offset<2; offset=offset+1)  begin
                        if(($signed(sram_58101x1056b_b2.mem[mem_idx][j-:BW_PER_ACT])===$signed(UP_2_golden_bank2[i][j-:BW_PER_ACT])-offset) ||
                           ($signed(sram_58101x1056b_b2.mem[mem_idx][j-:BW_PER_ACT])===$signed(UP_2_golden_bank2[i][j-:BW_PER_ACT])+offset)    )  begin
                            wrong = 0;
                        end
                    end
                    if(wrong==1)  begin
                        $display("You have wrong answer in sram B2 addr %d...", mem_idx);
                        $display("The %d pixel at channel %d: ",pixel ,23-j/(4*BW_PER_ACT));
                        $display("RTL: %b", sram_58101x1056b_b2.mem[mem_idx][j-:BW_PER_ACT]);
                        $display("GOL: %b", UP_2_golden_bank2[i][j-:BW_PER_ACT]);
                        $finish;
                    end
                    pixel = pixel + 1;
                end
            end

            pixel = 0;
            for(i=0; i<=58100; i=i+1)  begin
                mem_idx = i;
                for(j=CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1; j>0; j=j-BW_PER_ACT)  begin
                    if(pixel==4)  begin
                        pixel = 0;
                    end

                    wrong = 1;
                    for(offset=0; offset<2; offset=offset+1)  begin
                        if(($signed(sram_58101x1056b_b3.mem[mem_idx][j-:BW_PER_ACT])===$signed(UP_2_golden_bank3[i][j-:BW_PER_ACT])-offset) ||
                           ($signed(sram_58101x1056b_b3.mem[mem_idx][j-:BW_PER_ACT])===$signed(UP_2_golden_bank3[i][j-:BW_PER_ACT])+offset)    )  begin
                            wrong = 0;
                        end
                    end
                    if(wrong==1)  begin
                        $display("You have wrong answer in sram B3 addr %d...", mem_idx);
                        $display("The %d pixel at channel %d: ",pixel ,23-j/(4*BW_PER_ACT));
                        $display("RTL: %b", sram_58101x1056b_b3.mem[mem_idx][j-:BW_PER_ACT]);
                        $display("GOL: %b", UP_2_golden_bank3[i][j-:BW_PER_ACT]);
                        $finish;
                    end
                    pixel = pixel + 1;
                end
            end
            $display("UP_2 is correct :)\n");
        end

        CONV2:  begin
        $display("Checking CONV2...");
            pixel = 0;
            for(i=0; i<=58100; i=i+1)  begin
                mem_idx = i;
                for(j=CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1; j>=CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1-11*BW_PER_ACT; j=j-BW_PER_ACT)  begin
                    if(pixel==4)  begin
                        pixel = 0;
                    end

                    wrong = 1;
                    for(offset=0; offset<2; offset=offset+1)  begin
                        if(($signed(sram_58101x1056b_a0.mem[mem_idx][j-:BW_PER_ACT])===$signed(conv2_golden_bank0[i][j-:BW_PER_ACT])-offset) ||
                           ($signed(sram_58101x1056b_a0.mem[mem_idx][j-:BW_PER_ACT])===$signed(conv2_golden_bank0[i][j-:BW_PER_ACT])+offset)    )  begin
                            wrong = 0;
                        end
                    end
                    if(wrong==1)  begin
                        $display("You have wrong answer in sram A0 addr %d...", mem_idx);
                        $display("The %d pixel at channel %d: ",pixel ,23-j/(4*BW_PER_ACT));
                        $display("RTL: %b", sram_58101x1056b_a0.mem[mem_idx][j-:BW_PER_ACT]);
                        $display("GOL: %b", conv2_golden_bank0[i][j-:BW_PER_ACT]);
                        $finish;
                    end
                    pixel = pixel + 1;
                end
            end

            pixel = 0;
            for(i=0; i<=58100; i=i+1)  begin
                mem_idx = i;
                for(j=CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1; j>=CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1-11*BW_PER_ACT; j=j-BW_PER_ACT)  begin
                    if(pixel==4)  begin
                        pixel = 0;
                    end

                    wrong = 1;
                    for(offset=0; offset<2; offset=offset+1)  begin
                        if(($signed(sram_58101x1056b_a1.mem[mem_idx][j-:BW_PER_ACT])===$signed(conv2_golden_bank1[i][j-:BW_PER_ACT])-offset) ||
                           ($signed(sram_58101x1056b_a1.mem[mem_idx][j-:BW_PER_ACT])===$signed(conv2_golden_bank1[i][j-:BW_PER_ACT])+offset)    )  begin
                            wrong = 0;
                        end
                    end
                    if(wrong==1)  begin
                        $display("You have wrong answer in sram A1 addr %d...", mem_idx);
                        $display("The %d pixel at channel %d: ",pixel ,23-j/(4*BW_PER_ACT));
                        $display("RTL: %b", sram_58101x1056b_a1.mem[mem_idx][j-:BW_PER_ACT]);
                        $display("GOL: %b", conv2_golden_bank1[i][j-:BW_PER_ACT]);
                        $finish;
                    end
                    pixel = pixel + 1;
                end
            end

            pixel = 0;
            for(i=0; i<=58100; i=i+1)  begin
                mem_idx = i;
                for(j=CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1; j>=CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1-11*BW_PER_ACT; j=j-BW_PER_ACT)  begin
                    if(pixel==4)  begin
                        pixel = 0;
                    end

                    wrong = 1;
                    for(offset=0; offset<2; offset=offset+1)  begin
                        if(($signed(sram_58101x1056b_a2.mem[mem_idx][j-:BW_PER_ACT])===$signed(conv2_golden_bank2[i][j-:BW_PER_ACT])-offset) ||
                           ($signed(sram_58101x1056b_a2.mem[mem_idx][j-:BW_PER_ACT])===$signed(conv2_golden_bank2[i][j-:BW_PER_ACT])+offset)    )  begin
                            wrong = 0;
                        end
                    end
                    if(wrong==1)  begin
                        $display("You have wrong answer in sram A2 addr %d...", mem_idx);
                        $display("The %d pixel at channel %d: ",pixel ,23-j/(4*BW_PER_ACT));
                        $display("RTL: %b", sram_58101x1056b_a2.mem[mem_idx][j-:BW_PER_ACT]);
                        $display("GOL: %b", conv2_golden_bank2[i][j-:BW_PER_ACT]);
                        $finish;
                    end
                    pixel = pixel + 1;
                end
            end

            pixel = 0;
            for(i=0; i<=58100; i=i+1)  begin
                mem_idx = i;
                for(j=CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1; j>=CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1-11*BW_PER_ACT; j=j-BW_PER_ACT)  begin
                    if(pixel==4)  begin
                        pixel = 0;
                    end

                    wrong = 1;
                    for(offset=0; offset<2; offset=offset+1)  begin
                        if(($signed(sram_58101x1056b_a3.mem[mem_idx][j-:BW_PER_ACT])===$signed(conv2_golden_bank3[i][j-:BW_PER_ACT])-offset) ||
                           ($signed(sram_58101x1056b_a3.mem[mem_idx][j-:BW_PER_ACT])===$signed(conv2_golden_bank3[i][j-:BW_PER_ACT])+offset)    )  begin
                            wrong = 0;
                        end
                    end
                    if(wrong==1)  begin
                        $display("You have wrong answer in sram A3 addr %d...", mem_idx);
                        $display("The %d pixel at channel %d: ",pixel ,23-j/(4*BW_PER_ACT));
                        $display("RTL: %b", sram_58101x1056b_a3.mem[mem_idx][j-:BW_PER_ACT]);
                        $display("GOL: %b", conv2_golden_bank3[i][j-:BW_PER_ACT]);
                        $finish;
                    end
                    pixel = pixel + 1;
                end
            end
            $display("Correct :)\n");
        end

        default:  begin
            $finish;
        end
    endcase
    $finish;
end


//gate_sim
// initial begin 
//     $sdf_annotate("../syn/netlist/top_syn.sdf", top);
// end

endmodule





