module sram_read_addr_ctl(
    input clk,
    input rst_n,
    input [8:0] row,
    input [9:0] col,
    input [3:0] state,
    input fmap_end,
    input [6:0] fmap_idx_delay4,
    output reg [15:0] sram_raddr_a0,
    output reg [15:0] sram_raddr_a1,
    output reg [15:0] sram_raddr_a2,
    output reg [15:0] sram_raddr_a3,
    output reg [15:0] sram_raddr_b0,
    output reg [15:0] sram_raddr_b1,
    output reg [15:0] sram_raddr_b2,
    output reg [15:0] sram_raddr_b3,
    output reg [8:0]  sram_raddr_weight,
    output reg [8:0]  sram_raddr_bias,
    output reg [15:0] read_addr0_delay5,
    output reg [15:0] read_addr1_delay5,
    output reg [15:0] read_addr2_delay5,
    output reg [15:0] read_addr3_delay5
);

parameter IDLE = 0, PADDING = 1, CONV1 = 2, RES_1 = 3, RES_2 = 4, UP_1 = 5, UP_2 = 6, CONV2 = 7, FINISH = 8;
reg [15:0] read_addr0, read_addr1, read_addr2, read_addr3;
reg [15:0] read_addr0_delay, read_addr0_delay2, read_addr0_delay3, read_addr0_delay4;
reg [15:0] read_addr1_delay, read_addr1_delay2, read_addr1_delay3, read_addr1_delay4;
reg [15:0] read_addr2_delay, read_addr2_delay2, read_addr2_delay3, read_addr2_delay4;
reg [15:0] read_addr3_delay, read_addr3_delay2, read_addr3_delay3, read_addr3_delay4;
reg [8:0] temp_sram_raddr_weight;
reg [8:0] temp_sram_raddr_bias;

//SRAM_A raddr ctl
always @*  begin
    read_addr0 = ((col+1) >> 1) + ((row+1) >> 1)*321;  // (col+1)/2 + ((row+1)/2)*321
    read_addr1 = (col >> 1)     + ((row+1) >> 1)*321;  // (col)/2   + ((row+1)/2)*321
    read_addr2 = ((col+1) >> 1) + (row >> 1)*321;      // (col+1)/2 + (row/2)*321
    read_addr3 = (col >> 1)     + (row >> 1)*321;      // (col)/2   + (row/2)*321

    sram_raddr_a0 = read_addr0;
    sram_raddr_a1 = read_addr1;
    sram_raddr_a2 = read_addr2;
    sram_raddr_a3 = read_addr3;
end

always @(posedge clk)  begin
    if(~rst_n)  begin
        read_addr0_delay <= 0;
        // read_addr0_delay2 <= 0;
        read_addr0_delay3 <= 0;
        read_addr0_delay4 <= 0;
        read_addr0_delay5 <= 0;
        read_addr1_delay  <= 0;
        // read_addr1_delay2 <= 0;
        read_addr1_delay3 <= 0;
        read_addr1_delay4 <= 0;
        read_addr1_delay5 <= 0;
        read_addr2_delay  <= 0;
        // read_addr2_delay2 <= 0;
        read_addr2_delay3 <= 0;
        read_addr2_delay4 <= 0;
        read_addr2_delay5 <= 0;
        read_addr3_delay  <= 0;
        // read_addr3_delay2 <= 0;
        read_addr3_delay3 <= 0;
        read_addr3_delay4 <= 0;
        read_addr3_delay5 <= 0;
        sram_raddr_weight <= 0;
        sram_raddr_bias <= 0;
    end
    else  begin
        read_addr0_delay  <= read_addr0;
        // read_addr0_delay2 <= read_addr0_delay;
        read_addr0_delay3 <= read_addr0_delay;
        read_addr0_delay4 <= read_addr0_delay3;
        read_addr0_delay5 <= read_addr0_delay4;
        read_addr1_delay  <= read_addr1;
        // read_addr1_delay2 <= read_addr1_delay;
        read_addr1_delay3 <= read_addr1_delay;
        read_addr1_delay4 <= read_addr1_delay3;
        read_addr1_delay5 <= read_addr1_delay4;
        read_addr2_delay  <= read_addr2;
        // read_addr2_delay2 <= read_addr2_delay;
        read_addr2_delay3 <= read_addr2_delay;
        read_addr2_delay4 <= read_addr2_delay3;
        read_addr2_delay5 <= read_addr2_delay4;
        read_addr3_delay  <= read_addr3;
        // read_addr3_delay2 <= read_addr3_delay;
        read_addr3_delay3 <= read_addr3_delay;
        read_addr3_delay4 <= read_addr3_delay3;
        read_addr3_delay5 <= read_addr3_delay4;
        sram_raddr_weight <= temp_sram_raddr_weight;
        sram_raddr_bias <= temp_sram_raddr_bias;
    end
end

//SRAM_B raddr ctl
always@*  begin
    if(state==RES_2)  begin
        sram_raddr_b0 = read_addr0_delay3;
        sram_raddr_b1 = read_addr1_delay3;
        sram_raddr_b2 = read_addr2_delay3;
        sram_raddr_b3 = read_addr3_delay3;
    end
    else  begin
        sram_raddr_b0 = read_addr0;
        sram_raddr_b1 = read_addr1;
        sram_raddr_b2 = read_addr2;
        sram_raddr_b3 = read_addr3;
    end
end

//SRAM_weight raddr ctl
always @*  begin
    if(state==CONV1 || state==RES_1 || state==RES_2 || state==UP_1 || state==UP_2 || state==CONV2)  begin
        if(fmap_end)
            temp_sram_raddr_weight = sram_raddr_weight + 1;
        // else if(fmap_idx_delay4==24)  //state transfer to RES_1 -> set weight_raddr to 24
        //     temp_sram_raddr_weight = 24;
        else
            temp_sram_raddr_weight = sram_raddr_weight;
    end
    else  begin
        temp_sram_raddr_weight = 0;
    end
end

// always @(posedge clk)  begin
//     if(~rst_n)
//         sram_raddr_weight <= 0;
//     else
//         sram_raddr_weight <= temp_sram_raddr_weight;
// end

//SRAM_bias raddr ctl
always @*  begin
    if(state==CONV1 || state==RES_1 || state==RES_2 || state==UP_1 || state==UP_2 || state==CONV2)  begin
        if(fmap_end)
            temp_sram_raddr_bias = sram_raddr_bias + 1;
        // else if(fmap_idx_delay4==24)  //state transfer to RES_1 -> set bias_raddr to 24
        //     temp_sram_raddr_bias = 24;
        else
            temp_sram_raddr_bias = sram_raddr_bias;
    end
    else  begin
        temp_sram_raddr_bias = 0;
    end
end

// always @(posedge clk)  begin
//     if(~rst_n)
//         sram_raddr_bias <= 0;
//     else
//         sram_raddr_bias <= temp_sram_raddr_bias;
// end


endmodule