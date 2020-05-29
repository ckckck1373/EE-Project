module padding_addr_ctl(
    input clk,
    input rst_n,
    input [3:0] state,
    output reg [15:0] pad_addr,
    output reg pad_end,
    output reg sram_wen_a
);

parameter IDLE = 0, PADDING = 1;
reg [15:0] temp_pad_addr;
reg temp_row_pad, row_pad;
reg temp_col_pad, col_pad;
reg temp_sram_wen_a;

//pad_addr ctl
always@*  begin
    if(state==PADDING)  begin
        //jump addr
        if(pad_addr==320)
            temp_pad_addr = 14445;
        else if(pad_addr==14525)
            temp_pad_addr = 28890;
        else if(pad_addr==29050)
            temp_pad_addr = 57780;
        else if(pad_addr==58100)
            temp_pad_addr = 321;
        else if(pad_addr==14124)
            temp_pad_addr = 14766;
        else if(pad_addr==28569)
            temp_pad_addr = 29211;
        else if(pad_addr==57459)
            temp_pad_addr = 401;
        else if(pad_addr==14204)
            temp_pad_addr = 481;
        else if(pad_addr==28729)
            temp_pad_addr = 641;
        //last one
        else if(pad_addr==57779)
            temp_pad_addr = pad_addr;
        //regulate change
        else if(row_pad==1)
            temp_pad_addr = pad_addr + 1;
        else if(col_pad==1)
            temp_pad_addr = pad_addr + 321;
        else
            temp_pad_addr = pad_addr;
    end
    else  begin
        temp_pad_addr = pad_addr;
    end
end

//sram_wen_a ctl
always@*  begin
    if(state==PADDING)  begin
        if(pad_addr==80 || pad_addr==320 || pad_addr==14525 || pad_addr==58100 || pad_addr==14124 || pad_addr==57459 || pad_addr==14204)  begin
            temp_sram_wen_a = ~sram_wen_a;
        end
        else  begin
            temp_sram_wen_a = sram_wen_a;
        end
    end
    else  begin
        temp_sram_wen_a = sram_wen_a;
    end
end

//row_pad or col_pad ctl
always@*  begin
    if(state==PADDING)  begin
        if(pad_addr==58100)  begin
            temp_row_pad = 0;
            temp_col_pad = 1;
        end
        else  begin
            temp_row_pad = row_pad;
            temp_col_pad = col_pad;
        end
    end
    else  begin
        temp_row_pad = row_pad;
        temp_col_pad = col_pad;
    end
end

//pad_end ctl
always@*  begin
    if(pad_addr==57779)
        pad_end = 1;
    else
        pad_end = 0;
end


always@(posedge clk)  begin
if(~rst_n)  begin
    pad_addr <= 0;
    row_pad <= 1;
    col_pad <= 0;
    sram_wen_a <= 1;
end
else  begin
    pad_addr <= temp_pad_addr;
    row_pad  <= temp_row_pad;
    col_pad  <= temp_col_pad;
    sram_wen_a <= temp_sram_wen_a;
end
end

endmodule