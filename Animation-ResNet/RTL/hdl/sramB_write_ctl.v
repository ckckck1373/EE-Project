module sramB_write_ctl #(
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
    input [1:0] map_type_delay5,
    input [15:0] read_addr0_delay5,
    input [15:0] read_addr1_delay5,
    input [15:0] read_addr2_delay5,
    input [15:0] read_addr3_delay5,
    input [6:0] fmap_idx_delay5,
    input signed [BW_PER_ACT-1:0] LU_out,
    input signed [BW_PER_ACT-1:0] RU_out,
    input signed [BW_PER_ACT-1:0] LD_out,
    input signed [BW_PER_ACT-1:0] RD_out,
    input output_en,
    output reg sram_wen_a0,
    output reg sram_wen_a1,
    output reg sram_wen_a2,
    output reg sram_wen_a3,
    output reg sram_wen_b0,
    output reg sram_wen_b1,
    output reg sram_wen_b2,
    output reg sram_wen_b3,
    output reg [CH_NUM*ACT_PER_ADDR-1:0] sram_bytemask_b0,  //[95:0]
    output reg [CH_NUM*ACT_PER_ADDR-1:0] sram_bytemask_b1,  //[95:0]
    output reg [CH_NUM*ACT_PER_ADDR-1:0] sram_bytemask_b2,  //[95:0]
    output reg [CH_NUM*ACT_PER_ADDR-1:0] sram_bytemask_b3,  //[95:0]
    output reg [15:0] sram_waddr_b0,
    output reg [15:0] sram_waddr_b1,
    output reg [15:0] sram_waddr_b2,
    output reg [15:0] sram_waddr_b3,
    output reg [CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1:0] sram_wdata_b0,
    output reg [CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1:0] sram_wdata_b1,
    output reg [CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1:0] sram_wdata_b2,
    output reg [CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1:0] sram_wdata_b3,
    output pad_end
);
parameter IDLE = 0, PADDING = 1, CONV1 = 2, RES_1 = 3, RES_2 = 4, UP_1 = 5, UP_2 = 6, CONV2 = 7, FINISH = 8;
reg [6:0] number;
wire [CH_NUM*ACT_PER_ADDR-1:0] sram_bytemask_LU;
wire [CH_NUM*ACT_PER_ADDR-1:0] sram_bytemask_RU;
wire [CH_NUM*ACT_PER_ADDR-1:0] sram_bytemask_LD;
wire [CH_NUM*ACT_PER_ADDR-1:0] sram_bytemask_RD;

//for padding
wire [15:0] pad_waddr;
wire sram_wen_a;

//bytemask module call
//LU
write_LU_case bytemask_LU(
    .fmap_idx_delay5(number),
    .sram_bytemask(sram_bytemask_LU)
);
//RU
write_RU_case bytemask_RU(
    .fmap_idx_delay5(number),
    .sram_bytemask(sram_bytemask_RU)
);
//LD
write_LD_case bytemask_LD(
    .fmap_idx_delay5(number),
    .sram_bytemask(sram_bytemask_LD)
);
//RD
write_RD_case bytemask_RD(
    .fmap_idx_delay5(number),
    .sram_bytemask(sram_bytemask_RD)
);

//padding_addr_ctl
padding_addr_ctl pad_ctl(
    .clk(clk),
    .rst_n(rst_n),
    .state(state),
    .pad_addr(pad_waddr),
    .pad_end(pad_end),
    .sram_wen_a(sram_wen_a)
);

//--------------------Main---------------------
//SRAM write addr ctl
always @*  begin
    if(state==PADDING)  begin
        sram_waddr_b0 = pad_waddr;
        sram_waddr_b1 = pad_waddr; 
        sram_waddr_b2 = pad_waddr;
        sram_waddr_b3 = pad_waddr;
    end
    else if(state==CONV1 || state==RES_1 || state==RES_2 || state==CONV2)  begin
        sram_waddr_b0 = read_addr0_delay5;
        sram_waddr_b1 = read_addr1_delay5;
        sram_waddr_b2 = read_addr2_delay5;
        sram_waddr_b3 = read_addr3_delay5;
    end
    else if(state==UP_1 || state==UP_2)  begin
        if(fmap_idx_delay5[1:0]==0)  begin
            sram_waddr_b0 = read_addr1_delay5 + read_addr2_delay5;
            sram_waddr_b1 = sram_waddr_b0;
            sram_waddr_b2 = sram_waddr_b0;
            sram_waddr_b3 = sram_waddr_b0;
        end
        else if(fmap_idx_delay5[1:0]==1)  begin
            sram_waddr_b1 = read_addr1_delay5 + read_addr2_delay5;
            sram_waddr_b3 = read_addr1_delay5 + read_addr2_delay5;
            sram_waddr_b0 = sram_waddr_b1 + 1;
            sram_waddr_b2 = sram_waddr_b1 + 1;
        end
        else if(fmap_idx_delay5[1:0]==2)  begin
            sram_waddr_b2 = read_addr1_delay5 + read_addr2_delay5;
            sram_waddr_b3 = read_addr1_delay5 + read_addr2_delay5;
            sram_waddr_b0 = sram_waddr_b2 + 321;
            sram_waddr_b1 = sram_waddr_b2 + 321;

        end
        else  begin
            sram_waddr_b3 = read_addr1_delay5 + read_addr2_delay5;
            sram_waddr_b0 = sram_waddr_b3 + 322;
            sram_waddr_b1 = sram_waddr_b3 + 321;
            sram_waddr_b2 = sram_waddr_b3 + 1;
        end
    end
    else  begin
        sram_waddr_b0 = read_addr0_delay5;
        sram_waddr_b1 = read_addr1_delay5;
        sram_waddr_b2 = read_addr2_delay5;
        sram_waddr_b3 = read_addr3_delay5;
    end
end

//SRAM write data ctl
always @*  begin
    if(state==PADDING)  begin
        sram_wdata_b0 = 0;
        sram_wdata_b1 = 0;
        sram_wdata_b2 = 0;
        sram_wdata_b3 = 0;
    end
    else if(state==CONV1 || state==RES_1 || state==RES_2 || state==CONV2)  begin
        case(map_type_delay5)
            2'd0:  begin
                sram_wdata_b0 = {96{LU_out}};
                sram_wdata_b1 = {96{RU_out}};
                sram_wdata_b2 = {96{LD_out}};
                sram_wdata_b3 = {96{RD_out}};
            end

            2'd1:  begin
                sram_wdata_b0 = {96{RU_out}};
                sram_wdata_b1 = {96{LU_out}};
                sram_wdata_b2 = {96{RD_out}};
                sram_wdata_b3 = {96{LD_out}};
            end

            2'd2:  begin
                sram_wdata_b0 = {96{LD_out}};
                sram_wdata_b1 = {96{RD_out}};
                sram_wdata_b2 = {96{LU_out}};
                sram_wdata_b3 = {96{RU_out}};
            end

            default:  begin  //2'd3
                sram_wdata_b0 = {96{RD_out}};
                sram_wdata_b1 = {96{LD_out}};
                sram_wdata_b2 = {96{RU_out}};
                sram_wdata_b3 = {96{LU_out}};
            end
        endcase
    end
    else if(state==UP_1 || state==UP_2)  begin
        case(fmap_idx_delay5[1:0])
            2'd0:  begin
                sram_wdata_b0 = {96{LU_out}};
                sram_wdata_b1 = {96{RU_out}};
                sram_wdata_b2 = {96{LD_out}};
                sram_wdata_b3 = {96{RD_out}};
            end
            2'd1:  begin
                sram_wdata_b0 = {96{RU_out}};
                sram_wdata_b1 = {96{LU_out}};
                sram_wdata_b2 = {96{RD_out}};
                sram_wdata_b3 = {96{LD_out}};
            end
            2'd2:  begin
                sram_wdata_b0 = {96{LD_out}};
                sram_wdata_b1 = {96{RD_out}};
                sram_wdata_b2 = {96{LU_out}};
                sram_wdata_b3 = {96{RU_out}};
            end
            default:  begin  //2'd3
                sram_wdata_b0 = {96{RD_out}};
                sram_wdata_b1 = {96{LD_out}};
                sram_wdata_b2 = {96{RU_out}};
                sram_wdata_b3 = {96{LU_out}};
            end
        endcase
    end
    else  begin
        sram_wdata_b0 = {96{LU_out}};
        sram_wdata_b1 = {96{RU_out}};
        sram_wdata_b2 = {96{LD_out}};
        sram_wdata_b3 = {96{RD_out}};
    end
end

//SRAM_B write enable ctl  // 0:write mode
always@*  begin
    if(state==PADDING)  begin
        sram_wen_b0 = 0;
        sram_wen_b1 = 0;
        sram_wen_b2 = 0;
        sram_wen_b3 = 0;
    end
    else if(state==CONV1 || state==RES_2 || state==UP_2)  begin
        if(output_en==1)  begin
            sram_wen_b0 = 0;
            sram_wen_b1 = 0;
            sram_wen_b2 = 0;
            sram_wen_b3 = 0;
        end
        else begin
            sram_wen_b0 = 1;
            sram_wen_b1 = 1;
            sram_wen_b2 = 1;
            sram_wen_b3 = 1;
        end      
    end
    else  begin
        sram_wen_b0 = 1;
        sram_wen_b1 = 1;
        sram_wen_b2 = 1;
        sram_wen_b3 = 1;
    end
end

//SRAM_A write enable ctl  // 0:write mode
always@*  begin
    if(state==PADDING)  begin
        sram_wen_a0 = sram_wen_a;
        sram_wen_a1 = sram_wen_a;
        sram_wen_a2 = sram_wen_a;
        sram_wen_a3 = sram_wen_a;
    end
    else if(state==RES_1 || state==UP_1 || state==CONV2)  begin
        if(output_en==1)  begin
            sram_wen_a0 = 0;
            sram_wen_a1 = 0;
            sram_wen_a2 = 0;
            sram_wen_a3 = 0;
        end
        else begin
            sram_wen_a0 = 1;
            sram_wen_a1 = 1;
            sram_wen_a2 = 1;
            sram_wen_a3 = 1;
        end      
    end
    else  begin
        sram_wen_a0 = 1;
        sram_wen_a1 = 1;
        sram_wen_a2 = 1;
        sram_wen_a3 = 1;
    end
end

//SRAM bytemask ctl  //0: addr can be written
always @*  begin
    if(state==PADDING)  begin
        number = fmap_idx_delay5;
        sram_bytemask_b0 = 0;
        sram_bytemask_b1 = 0;
        sram_bytemask_b2 = 0;
        sram_bytemask_b3 = 0;
    end
    else if(state==CONV1 || state==RES_1 || state==RES_2 || state==CONV2)  begin
        number = fmap_idx_delay5;
        case(map_type_delay5)
            2'd0:  begin
                sram_bytemask_b0 = sram_bytemask_RD;
                sram_bytemask_b1 = sram_bytemask_LD;
                sram_bytemask_b2 = sram_bytemask_RU;
                sram_bytemask_b3 = sram_bytemask_LU;
            end
            2'd1:  begin
                sram_bytemask_b0 = sram_bytemask_LD;
                sram_bytemask_b1 = sram_bytemask_RD;
                sram_bytemask_b2 = sram_bytemask_LU;
                sram_bytemask_b3 = sram_bytemask_RU;
            end
            2'd2:  begin
                sram_bytemask_b0 = sram_bytemask_RU;
                sram_bytemask_b1 = sram_bytemask_LU;
                sram_bytemask_b2 = sram_bytemask_RD;
                sram_bytemask_b3 = sram_bytemask_LD;
            end
            default:  begin  //2'd3
                sram_bytemask_b0 = sram_bytemask_LU;
                sram_bytemask_b1 = sram_bytemask_RU;
                sram_bytemask_b2 = sram_bytemask_LD;
                sram_bytemask_b3 = sram_bytemask_RD;
            end
        endcase
    end
    else if(state==UP_1 || state==UP_2)  begin
        number = fmap_idx_delay5 >> 2;
        case(fmap_idx_delay5[1:0])
            2'd0:  begin
                sram_bytemask_b0 = sram_bytemask_RD;
                sram_bytemask_b1 = sram_bytemask_RD;
                sram_bytemask_b2 = sram_bytemask_RD;
                sram_bytemask_b3 = sram_bytemask_RD;
            end
            2'd1:  begin
                sram_bytemask_b0 = sram_bytemask_LD;
                sram_bytemask_b1 = sram_bytemask_LD;
                sram_bytemask_b2 = sram_bytemask_LD;
                sram_bytemask_b3 = sram_bytemask_LD;
            end
            2'd2:  begin
                sram_bytemask_b0 = sram_bytemask_RU;
                sram_bytemask_b1 = sram_bytemask_RU;
                sram_bytemask_b2 = sram_bytemask_RU;
                sram_bytemask_b3 = sram_bytemask_RU;
            end
            default:  begin  //2'd3
                sram_bytemask_b0 = sram_bytemask_LU;
                sram_bytemask_b1 = sram_bytemask_LU;
                sram_bytemask_b2 = sram_bytemask_LU;
                sram_bytemask_b3 = sram_bytemask_LU;
            end
        endcase
    end
    else  begin
        number = fmap_idx_delay5;
        sram_bytemask_b0 = sram_bytemask_RD;
        sram_bytemask_b1 = sram_bytemask_LD;
        sram_bytemask_b2 = sram_bytemask_RU;
        sram_bytemask_b3 = sram_bytemask_LU;
    end
end

endmodule