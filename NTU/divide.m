function [train,val,test] = divide(x)
    j = 1;
    k = 1;
    l = 1;
    str_train = ["P002","P003","P004","P005","P006","P007","P008","P009","P010","P011","P012","P013","P015","P017","P018"];
    str_val = ["P001","P014","P016"];
    str_test = ["P002","P019","P020"];
    for i = 3:length(x)
        if(contains(x(i).name,str_train))
            train(j).name = x(i).name;
            j = j + 1;
        elseif(contains(x(i).name,str_val))
            val(k).name = x(i).name;
            k = k + 1;
        elseif(contains(x(i).name,str_test))
            test(l).name = x(i).name;
            l = l + 1;
        end
        
    end
end
