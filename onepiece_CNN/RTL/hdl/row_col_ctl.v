module row_col_ctl(
    input clk,
    input rst_n,
    input [3:0] state,
    input fmap_end,
    input [6:0] fmap_idx_delay4,
    output reg [8:0] row,  // total row: 720/2 = 360
    output reg [9:0] col   // total col: 1280/2 = 640
);

parameter IDLE = 0, PADDING = 1, CONV1 = 2, RES_1 = 3, RES_2 = 4, UP_1 = 5, UP_2 = 6, CONV2 = 7, FINISH = 8; // CONV1:(320*180, 3)->(320*180, 24) 
reg [8:0] n_row;
reg [9:0] n_col;

always @*  begin
  // col ctl
    if(state==CONV1 || state==RES_1 || state==RES_2)  begin
        if(fmap_end)
            n_col = 0;
        else if(col==159)
            n_col = 0;
        else if(fmap_idx_delay4==24)  //state transfer -> reset col to 0
            n_col = 0;
        else
            n_col = col + 1;
    end
    else if(state==UP_1)  begin
        if(fmap_end)
            n_col = 0;
        else if(col==159)
            n_col = 0;
        else if(fmap_idx_delay4==96)  //state transfer -> reset col to 0
            n_col = 0;
        else
            n_col = col + 1;
    end
    else if(state==UP_2)  begin
        if(fmap_end)
            n_col = 0;
        else if(col==319)
            n_col = 0;
        else if(fmap_idx_delay4==96)  //state transfer -> reset col to 0
            n_col = 0;
        else
            n_col = col + 1;
    end
    else if(state==CONV2)  begin
        if(fmap_end)
            n_col = 0;
        else if(col==639)
            n_col = 0;
        else if(fmap_idx_delay4==24)  //state transfer -> reset col to 0
            n_col = 0;
        else
            n_col = col + 1;
    end
    else  begin
        n_col = 0;
    end

  // row ctl
    if(state==CONV1 || state==RES_1 || state==RES_2)  begin
        if(fmap_end)
            n_row = 0;
        else if(col==159)
            n_row = row + 1;
        else if(fmap_idx_delay4==24)  //state transfer -> reset row to 0
            n_row = 0;
        else
            n_row = row;
    end
    else if(state==UP_1)  begin
        if(fmap_end)
            n_row = 0;
        else if(col==159)
            n_row = row + 1;
        else if(fmap_idx_delay4==96)  //state transfer -> reset row to 0
            n_row = 0;
        else
            n_row = row;
    end
    else if(state==UP_2)  begin
        if(fmap_end)
            n_row = 0;
        else if(col==319)
            n_row = row + 1;
        else if(fmap_idx_delay4==96)  //state transfer -> reset row to 0
            n_row = 0;
        else
            n_row = row;
    end
    else if(state==CONV2)  begin
        if(fmap_end)
            n_row = 0;
        else if(col==639)
            n_row = row + 1;
        else if(fmap_idx_delay4==24)  //state transfer -> reset row to 0
            n_row = 0;
        else
            n_row = row;
    end
    else  begin
        n_row = 0;
    end
end

always @(posedge clk)  begin
    if(~rst_n)  begin
        row <= 0;
        col <= 0;
    end
    else  begin
        row <= n_row;
        col <= n_col;
    end
end

endmodule