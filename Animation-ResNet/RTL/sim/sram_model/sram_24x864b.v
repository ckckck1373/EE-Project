module sram_24x864b #(       //for weight
parameter WEIGHT_PER_ADDR = 216,
parameter BW_PER_WEIGHT = 8
)
(
input clk,
input csb,  //chip enable
input wsb,  //write enable
input [WEIGHT_PER_ADDR*BW_PER_WEIGHT-1:0] wdata, //write data
input [8:0] waddr, //write address
input [8:0] raddr, //read address

output reg [WEIGHT_PER_ADDR*BW_PER_WEIGHT-1:0] rdata //read data 864 bits
);
/*
Data location
/////////////////////
addr 0~23: conv1_w(24) (3 -> 24)

/////////////////////
*/
reg [WEIGHT_PER_ADDR*BW_PER_WEIGHT-1:0] mem [0:410];

always @(negedge clk) begin
    if(~csb && ~wsb)
        mem[waddr] <= wdata;
end

always @(negedge clk) begin
    if(~csb)
        rdata <= mem[raddr];
end


task load_param(
    input integer index,
    input [WEIGHT_PER_ADDR*BW_PER_WEIGHT-1:0] param_input
);
    mem[index] = param_input;
endtask

endmodule