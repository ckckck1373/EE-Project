module sram_24x4b #(       //for bias
parameter BIAS_PER_ADDR = 1,
parameter BW_PER_BIAS = 8
)
(
input clk,
input csb,  //chip enable
input wsb,  //write enable
input [BIAS_PER_ADDR*BW_PER_BIAS-1:0] wdata, //write data
input [8:0] waddr, //write address
input [8:0] raddr, //read address

output reg [BIAS_PER_ADDR*BW_PER_BIAS-1:0] rdata //read data 4 bits
);
/*
Data location
/////////////////////
addr 0~23: conv1_b(24)

/////////////////////
*/
reg [BIAS_PER_ADDR*BW_PER_BIAS-1:0] mem [0:410];

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
    input [BIAS_PER_ADDR*BW_PER_BIAS-1:0] param_input
);
    mem[index] = param_input;
endtask

endmodule