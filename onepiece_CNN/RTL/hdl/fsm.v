module fsm(
    input clk,
    input rst_n,
    input enable,
    input [6:0] fmap_idx_delay4,
    input pad_end,
    output reg [3:0] state
);

parameter IDLE = 0, PADDING = 1, CONV1 = 2, RES_1 = 3, RES_2 = 4, UP_1 = 5, UP_2 = 6, CONV2 = 7, FINISH = 8;

reg [3:0] n_state;
reg [3:0] res_count, temp_res_count;

always @*  begin
    case(state)
        IDLE:   n_state = (enable==1)? PADDING : IDLE;
        PADDING:n_state = (pad_end==1)? CONV1 : PADDING;
        CONV1:  n_state = (fmap_idx_delay4==24)? RES_1 : CONV1;
        RES_1:  n_state = (fmap_idx_delay4==24)? RES_2 : RES_1;
        RES_2:  begin
            // if(fmap_idx_delay4==24)
            //     n_state = FINISH;
            // else
            //     n_state = RES_2;
            if(res_count==8 && fmap_idx_delay4==24)  begin
                n_state = UP_1;
            end
            else  if(fmap_idx_delay4==24)  begin
                n_state = RES_1;
            end
            else  begin
                n_state = RES_2;
            end
        end
        UP_1:   n_state = (fmap_idx_delay4==96)? UP_2 : UP_1;
        UP_2:   n_state = (fmap_idx_delay4==96)? CONV2 : UP_2;
        CONV2:  n_state = (fmap_idx_delay4==3)? FINISH : CONV2;
        FINISH: n_state = FINISH;
        default: n_state = IDLE;
    endcase
end

//RESBLOCK count
always@*  begin
    if(fmap_idx_delay4==24)  begin
        temp_res_count = res_count + 1;
    end
    else  begin
        temp_res_count = res_count;
    end
end

always @(posedge clk)  begin
    if(~rst_n)  begin
        state <= IDLE;
        res_count <= 0;
    end
    else  begin
        state <= n_state;
        res_count <= temp_res_count;
    end
end


endmodule