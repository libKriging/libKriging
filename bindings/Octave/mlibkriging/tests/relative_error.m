function err = relative_error(x, y)
     if ~isequal(size(x), size(y))
        error("Inconsistent vector sizes");
     end
     x_norm = norm(x);
     y_norm = norm(y);
     if x_norm > 0 || y_norm > 0
         diff_norm = norm(x - y);
         err = diff_norm / max(x_norm, y_norm);
     else
        err = 0;
     end
end