package utilities;

import java.util.function.BiConsumer;
import java.util.function.BiFunction;

public class StringUtilities {

    public static void forEachPair(String[] options, BiConsumer<String, String> function) {
        if(options.length % 2 != 0) {
            throw new IllegalArgumentException("options is not correct length, must be key-value pairs");
        }
        for(int i = 0; i < options.length; i += 2) {
            String key = options[i];
            String value = options[i + 1];
            function.accept(key, value);
        }
    }
}
