module counter(
    input clk,
    input rst_n,
    input [3:0] state,
    output reg [17:0] count,
    output reg fmap_end,
    output reg [6:0] fmap_idx_delay4,
    output reg [6:0] fmap_idx_delay5,
    output reg output_en
);
parameter IDLE = 0, PADDING = 1, CONV1 = 2, RES_1 = 3, RES_2 = 4, UP_1 = 5, UP_2 = 6, CONV2 = 7, FINISH = 8;
reg [17:0] n_count;
reg [6:0] temp_fmap_idx, fmap_idx, fmap_idx_delay, fmap_idx_delay2, fmap_idx_delay3;
reg [2:0] pipeline_count, n_pipeline_count;


always @*  begin
    if(state==CONV1 || state==RES_1 || state==RES_2 || state==CONV2)  begin
        if((state==CONV1 || state==RES_1 || state==RES_2) && (count==14399))  begin
            n_count = 0;
            fmap_end = 1;
            temp_fmap_idx = fmap_idx + 1;
        end
        else if((state==CONV2) && (count==230399))  begin
            n_count = 0;
            fmap_end = 1;
            temp_fmap_idx = fmap_idx + 1;
        end
        else if(fmap_idx==24)  begin
            n_count = count + 1;  //don't care
            fmap_end = 0;         //don't care
            temp_fmap_idx = 127;  //don't care, just can't be 24 to make sure that 24 appears in a cycle for state transfer
        end
        else if(fmap_idx_delay4==24)  begin  //state transfer -> reset count/fmap_idx to 0
            n_count = 0;
            fmap_end = 0;
            temp_fmap_idx = 0;
        end
        else  begin
            n_count = count + 1;
            fmap_end = 0;
            temp_fmap_idx = fmap_idx;
        end
    end
    else if(state==UP_1 || state==UP_2)  begin
        if(state==UP_1 && count==14399)  begin
            n_count = 0;
            fmap_end = 1;
            temp_fmap_idx = fmap_idx + 1;
        end
        else if(state==UP_2 && count==57599)  begin
            n_count = 0;
            fmap_end = 1;
            temp_fmap_idx = fmap_idx + 1;
        end
        else if(fmap_idx==96)  begin
            n_count = count + 1;  //don't care
            fmap_end = 0;         //don't care
            temp_fmap_idx = 127;  //don't care, just can't be 96 to make sure that 96 appears in a cycle for state transfer
        end
        else if(fmap_idx_delay4==96)  begin  //state transfer -> reset count/fmap_idx to 0
            n_count = 0;
            fmap_end = 0;
            temp_fmap_idx = 0;
        end
        else  begin
            n_count = count + 1;
            fmap_end = 0;
            temp_fmap_idx = fmap_idx;
        end
    end
    else  begin
        n_count = 0;
        fmap_end = 0;
        temp_fmap_idx = 0;
    end
end


always @(posedge clk)  begin
    if(~rst_n)  begin
        count <= 0;
        fmap_idx <= 0;
        fmap_idx_delay <= 0;
        // fmap_idx_delay2 <= 0;
        fmap_idx_delay3 <= 0;
        fmap_idx_delay4 <= 0;
        fmap_idx_delay5 <= 0;
    end
    else  begin
        count <= n_count;
        fmap_idx <= temp_fmap_idx;
        fmap_idx_delay <= fmap_idx;
        // fmap_idx_delay2 <= fmap_idx_delay;
        fmap_idx_delay3 <= fmap_idx_delay;
        fmap_idx_delay4 <= fmap_idx_delay3;
        fmap_idx_delay5 <= fmap_idx_delay4;
    end
end

//output_en ctl
always@*  begin
    if(state==CONV1 || state==RES_1 || state==RES_2 || state==CONV2)  begin
        if(fmap_idx_delay4==24)  begin  //state transfer -> reset pipeline_count to 0
            n_pipeline_count = 0;
            output_en = 1;
        end
        else if(pipeline_count==4)  begin
            n_pipeline_count = pipeline_count;
            output_en = 1;
        end
        else  begin
            n_pipeline_count = pipeline_count + 1;
            output_en = 0;
        end
    end
    else if(state==UP_1 || state==UP_2)  begin
        if(fmap_idx_delay4==96)  begin  //state transfer -> reset pipeline_count to 0
            n_pipeline_count = 0;
            output_en = 1;
        end
        else if(pipeline_count==4)  begin
            n_pipeline_count = pipeline_count;
            output_en = 1;
        end
        else  begin
            n_pipeline_count = pipeline_count + 1;
            output_en = 0;
        end
    end
    else  begin
        n_pipeline_count = 0;
        output_en = 0;
    end
end

always @(posedge clk)  begin
    if(~rst_n)
        pipeline_count <= 0;
    else
        pipeline_count <= n_pipeline_count;
end

endmodule