module sram_58101x1056b #(     //for activation
parameter CH_NUM = 24,
parameter ACT_PER_ADDR = 4,
parameter BW_PER_ACT = 16
)
(
input clk,
input [CH_NUM*ACT_PER_ADDR-1:0] bytemask,  //96 bits
input csb,  //chip enable
input wsb,  //write enable
input [CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1:0] wdata, //write data 1056 bits
input [15:0] waddr, //write address
input [15:0] raddr, //read address

output reg [CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1:0] rdata //read data 1056 bits
);

reg [CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1:0] mem [0:58100];
wire [CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1:0] bit_mask;




assign bit_mask = { {16{bytemask[95]}}, {16{bytemask[94]}}, {16{bytemask[93]}}, {16{bytemask[92]}}, {16{bytemask[91]}}, {16{bytemask[90]}}, 
                    {16{bytemask[89]}}, {16{bytemask[88]}}, {16{bytemask[87]}}, {16{bytemask[86]}}, {16{bytemask[85]}}, {16{bytemask[84]}}, {16{bytemask[83]}}, {16{bytemask[82]}}, {16{bytemask[81]}}, {16{bytemask[80]}},
                    {16{bytemask[79]}}, {16{bytemask[78]}}, {16{bytemask[77]}}, {16{bytemask[76]}}, {16{bytemask[75]}}, {16{bytemask[74]}}, {16{bytemask[73]}}, {16{bytemask[72]}}, {16{bytemask[71]}}, {16{bytemask[70]}},
                    {16{bytemask[69]}}, {16{bytemask[68]}}, {16{bytemask[67]}}, {16{bytemask[66]}}, {16{bytemask[65]}}, {16{bytemask[64]}}, {16{bytemask[63]}}, {16{bytemask[62]}}, {16{bytemask[61]}}, {16{bytemask[60]}},
                    {16{bytemask[59]}}, {16{bytemask[58]}}, {16{bytemask[57]}}, {16{bytemask[56]}}, {16{bytemask[55]}}, {16{bytemask[54]}}, {16{bytemask[53]}}, {16{bytemask[52]}}, {16{bytemask[51]}}, {16{bytemask[50]}},
                    {16{bytemask[49]}}, {16{bytemask[48]}}, {16{bytemask[47]}}, {16{bytemask[46]}}, {16{bytemask[45]}}, {16{bytemask[44]}}, {16{bytemask[43]}}, {16{bytemask[42]}}, {16{bytemask[41]}}, {16{bytemask[40]}},
                    {16{bytemask[39]}}, {16{bytemask[38]}}, {16{bytemask[37]}}, {16{bytemask[36]}}, {16{bytemask[35]}}, {16{bytemask[34]}}, {16{bytemask[33]}}, {16{bytemask[32]}}, {16{bytemask[31]}}, {16{bytemask[30]}},
                    {16{bytemask[29]}}, {16{bytemask[28]}}, {16{bytemask[27]}}, {16{bytemask[26]}}, {16{bytemask[25]}}, {16{bytemask[24]}}, {16{bytemask[23]}}, {16{bytemask[22]}}, {16{bytemask[21]}}, {16{bytemask[20]}},
                    {16{bytemask[19]}}, {16{bytemask[18]}}, {16{bytemask[17]}}, {16{bytemask[16]}}, {16{bytemask[15]}}, {16{bytemask[14]}}, {16{bytemask[13]}}, {16{bytemask[12]}}, {16{bytemask[11]}}, {16{bytemask[10]}},
                    {16{bytemask[9]}}, {16{bytemask[8]}}, {16{bytemask[7]}}, {16{bytemask[6]}}, {16{bytemask[5]}}, {16{bytemask[4]}}, {16{bytemask[3]}}, {16{bytemask[2]}}, {16{bytemask[1]}}, {16{bytemask[0]}}
                  };

always @(negedge clk) begin
    if(~csb && ~wsb) begin
        mem[waddr] <= (wdata & ~(bit_mask)) | (mem[waddr] & bit_mask);
    end
end

always @(negedge clk) begin
    if(~csb) begin
        rdata <= mem[raddr];
    end
end

task load_param(
    input integer index,
    input [CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1:0] param_input
);
    mem[index] = param_input;
endtask

endmodule