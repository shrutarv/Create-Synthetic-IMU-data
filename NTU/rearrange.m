function data = rearrange(x)
    x_value = {x.x};
    y_value = {x.y};
    z_value = {x.z};
    j = 1;
    for i = 1:25
        data(j) = x_value{i};
        data(j+1) = y_value{i};
        data(j+2) = z_value{i};
        j = j+3;
             
end