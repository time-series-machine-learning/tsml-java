package utilities;

import java.util.LinkedList;
import java.util.List;

public class StringUtilities {

    public static String join(String separator, String... parts) {
        StringBuilder list = new StringBuilder();
        for(int i = 0; i < parts.length - 1; i++){
            list.append(parts[i]);
            list.append(separator);
        }
        list.append(parts[parts.length - 1]);
        return list.toString();
    }
}
