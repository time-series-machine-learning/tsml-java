package tsml.classifiers.distance_based.utils.strings;

import experiments.ClassifierLists;
import experiments.Experiments;
import java.time.Duration;
import org.apache.commons.lang3.StringUtils;
import tsml.classifiers.EnhancedAbstractClassifier;
import tsml.classifiers.distance_based.utils.collections.params.ParamHandler;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.OptionHandler;
import weka.core.Utils;

import java.io.File;
import java.util.*;
import java.util.function.BiConsumer;
import java.util.function.Consumer;
import java.util.function.Function;

public class StrUtils {

    public static String toString(double[][] matrix) {
        StringBuilder builder = new StringBuilder();
        boolean first = true;
        for(double[] row : matrix) {
            if(first) {
                first = false;
            } else {
                builder.append(System.lineSeparator());
            }
            builder.append(Arrays.toString(row));
        }
        return builder.toString();
    }

    public static String durationToHmsString(Duration duration) {
        return duration.toString().substring(2)
            //            .replaceAll("(\\d[HMS])(?!$)", "$1 ")
            .toLowerCase();
    }

    public static String[] extractAmountAndUnit(String str) {
        str = str.trim();
        StringBuilder amount = new StringBuilder();
        StringBuilder unit = new StringBuilder();
        char[] chars = str.toCharArray();
        int i = 0;
        boolean digitsEnded = false;
        for(;i < chars.length; i++) {
            char c = chars[i];
            if(!Character.isDigit(c)) {
                digitsEnded = true;
            }
            if(digitsEnded) {
                unit.append(c);
            } else {
                amount.append(c);
            }
        }
        return new String[] {amount.toString(), unit.toString()};
    }

    public static String depluralise(String str) {
        if(str.endsWith("s")) {
            str = str.substring(0, str.length() - 1);
        }
        return str;
    }

    public static String extractNameAndParams(Classifier classifier) {
        String params;
        if(classifier instanceof ParamHandler) {
            params = ((ParamHandler) classifier).getParams().toString();
        } else if(classifier instanceof EnhancedAbstractClassifier) {
            params = ((EnhancedAbstractClassifier) classifier).getParameters();
        } else {
            return classifier.toString();
        }
        return classifier.toString() + " " + params;
    }

    public static String joinOptions(List<String> options) {
        // todo use view
        return Utils.joinOptions(options.toArray(new String[0]));
    }

    public static String join(String separator, String... parts) {
        return join(separator, Arrays.asList(parts));
    }

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

    public static String join(String separator, double... values) {
        String[] strings = new String[values.length];
        for (int i = 0; i < values.length; i++) {
            strings[i] = String.valueOf(values[i]);
        }
        return join(separator, strings);
    }

    public static HashMap<String, String> pairValuesToMap(String... pairValues) {
        HashMap<String, String> map = new HashMap<>();
        forEachPair(pairValues, map::put);
        return map;
    }

    public static boolean equalPairs(String[] a, String[] b) {
        HashMap<String, String> mapA = pairValuesToMap(a);
        HashMap<String, String> mapB = pairValuesToMap(b);
        for(Map.Entry<String, String> entry : mapA.entrySet()) {
            String valueA = entry.getValue();
            String keyA = entry.getKey();
            String valueB = mapB.get(keyA);
            if(!valueA.equals(valueB)) {
                return false;
            }
        }
        return true;
    }

    public static <A> void setOption(String[] options, String flag, Consumer<A> setter, Function<String, A> parser) throws
                                                                                                                    Exception {
        flag = unflagify(flag);
        String value = Utils.getOption(flag, options);
        if(value.length() > 0) {
            A parsed = parser.apply(value);
            setter.accept(parsed);
        }
    }


    public static void setFlag(String[] options, String flag, Consumer<Boolean> setter) throws
                                                                                      Exception {
        flag = unflagify(flag);
        setter.accept(Utils.getFlag(flag, options));
    }

    public static boolean isOption(String flag, List<String> options) {
        // todo use view instead
        return isOption(flag, options.toArray(new String[0]));
    }

    public static boolean isOption(String flag, String[] options) {
        try {
            flag = unflagify(flag);
            int i = Utils.getOptionPos(flag, options);
            if(i < 0 || i + 1 >= options.length) {
                return false;
            }
            String option = options[i + 1];
            if(isFlag(option)) {
                return false;
            }
            return true;
        } catch (Exception e) {
            return false;
        }
    }

    public static void updateOptions(OptionHandler optionHandler, String[] options) throws
                                                                                    Exception {
        String[] current = optionHandler.getOptions();
        String[] next = uniteOptions(options, current);
        optionHandler.setOptions(next);
    }

    public static String[] uniteOptions(String[] next, String[] current) throws
                                                                                                   Exception {
        return uniteOptions(next, current, true);
    }

    public static String[] uniteOptions(String[] next, String[] current, boolean flagsFromCurrent) throws
                                                                         Exception {
        ArrayList<String> list = new ArrayList<>();
        for(int i = 0; i < current.length; i++) {
            String flag = current[i];
            flag = unflagify(flag);
            if(isOption(flag, current)) {
                String nextOption = Utils.getOption(flag, next);
                if(nextOption.length() > 0) {
                    addOption(flag, list, nextOption);
                } else {
                    String currentOption = Utils.getOption(flag, current);
                    addOption(flag, list, currentOption);
                }
                i++;
            } else {
                if(Utils.getFlag(flag, next) || flagsFromCurrent) {
                    addFlag(flag, list);
                } else {
                    // don't add the flag
                }
            }

        }
        return list.toArray(new String[0]);
    }

    public static <A> void setOption(String[] options, String flag, Consumer<A> setter, Class<? super A> baseClass) throws
                                                                Exception {
        flag = unflagify(flag);
        String option = Utils.getOption(flag, options);
        while(option.length() != 0) {
            A result = fromOptionValue(option, baseClass);
            setter.accept(result);
            option = Utils.getOption(flag, options);
        }
    }


    public static AbstractClassifier classifierFromClassifierLists(String name) {
        Experiments.ExperimentalArguments experimentalArguments = new Experiments.ExperimentalArguments();
        experimentalArguments.foldId = 0;
        experimentalArguments.classifierName = name;
        Classifier classifier = ClassifierLists.setClassifier(experimentalArguments);
        return (AbstractClassifier) classifier;
    }

    public static AbstractClassifier classifierFromClassName(String name) {
        try {
            return fromOptionValue(name, AbstractClassifier.class);
        } catch (Exception e) {
            throw new IllegalArgumentException(e);
        }
    }

    public static EnhancedAbstractClassifier enhancedClassifierFromClassName(String name) {
        try {
            return fromOptionValue(name, EnhancedAbstractClassifier.class);
        } catch (Exception e) {
            throw new IllegalArgumentException(e);
        }
    }

    public static AbstractClassifier classifierFromString(String name) {
        try {
            return classifierFromClassName(name);
        } catch (Exception e) {
            try {
                return classifierFromClassifierLists(name);
            } catch(Exception f) {
                throw new IllegalArgumentException("unknown classifier: " + name);
            }
        }
    }


    public static EnhancedAbstractClassifier enhancedClassifierFromClassifierLists(String name) {
        Experiments.ExperimentalArguments experimentalArguments = new Experiments.ExperimentalArguments();
        experimentalArguments.foldId = 0;
        experimentalArguments.classifierName = name;
        Classifier classifier = ClassifierLists.setClassifier(experimentalArguments);
        if(!(classifier instanceof EnhancedAbstractClassifier)) {
            throw new IllegalStateException();
        }
        return (EnhancedAbstractClassifier) classifier;
    }

    public static EnhancedAbstractClassifier enhancedClassifierFromString(String name) {
        try {
            return enhancedClassifierFromClassName(name);
        } catch (Exception e) {
            try {
                return enhancedClassifierFromClassifierLists(name);
            } catch(Exception f) {
                throw new IllegalArgumentException("unknown classifier: " + name);
            }
        }
    }

    public static Object fromOptionValue(String option) throws
                                                        Exception {
        return fromOptionValue(option, Object.class);
    }

    public static <A> A cast(Object value, Function<String, A> fromString) {
        if(value instanceof String) {
            value = fromString.apply((String) value);
        }
        return (A) value;
    }

    public static <A> A fromOptionValue(String option, Class<? super A> baseClass) throws Exception {
        if(option.equals("null")) {
            return null;
        }
        String[] parts = Utils.splitOptions(option);
        if(parts.length == 0) {
            throw new Exception("Invalid option: " + option);
        }
        if(parts.length == 1) {
            try {
                return (A) parts[0];
            } catch(ClassCastException e) {
                throw new IllegalArgumentException("cannot cast " + parts[0] + " to " + baseClass.getName());
            }
        }
        String className = parts[0];
        parts[0] = "";
        return forName(baseClass,
                       className,
                       parts);
    }

    public static <A> A forName(Class<? super A> classType,
                                String className) throws
                                                  Exception {
        return forName(classType, className, new String[0]);
    }

    public static <A> A forName(Class<? super A> classType,
                                 String className,
                                 String[] options) throws Exception {
        // CANNOT HANDLE STRINGS WITH SPACES!
        if(className.equals(String.class.getName())) {
            if(options.length != 1) {
                throw new IllegalArgumentException("can only construct string from 1 option");
            }
            try {
                A result = (A) options[0];
                return result;
            } catch (ClassCastException e) {
                throw new Exception("not castable to string");
            }
        }
        Class<?> c = null;
        try {
            c = Class.forName(className);
        } catch (Exception ex) {
            int index = className.indexOf(".");
            if(index < 0) {
                throw new Exception("Can't find class called: " + className);
            }
            className = className.substring(index + 1);
            try {
                c = Class.forName(className);
            } catch (Exception e) {
                throw new Exception("couldn't find degraded class name" + className);
            }
        }
        if (!classType.isAssignableFrom(c)) {
            throw new Exception(classType.getName() + " is not assignable from "
                                + className);
        }
        Object o = c.newInstance();
        if ((o instanceof OptionHandler)
            && (options != null)) {
            ((OptionHandler)o).setOptions(options);
//            Utils.checkForRemainingOptions(options);
        }
        try {
            A result = (A) o;
            return result;
        } catch (ClassCastException e) {
            throw new Exception(className + " not castable to "
                                + classType.getName());
        }
    }

    public static String toOptionValue(Object value) {
        // CANNOT HANDLE STRINGS WITH SPACES!
        String str;
        if(value == null) {
            str = "null";
        }
        else if(value instanceof Integer ||
           value instanceof Double ||
           value instanceof Float ||
           value instanceof Byte ||
           value instanceof Short ||
           value instanceof Long ||
           value instanceof Character ||
           value instanceof Boolean) {
            str = "\"" + String.valueOf(value) + "\"";
        } else if(value instanceof String) {
            str = "\"\\\"" + value + "\\\"\"";
        } else {
            Class<?> classValue;
            if(value instanceof Class<?>) {
                classValue = (Class<?>) value;
            } else {
                classValue = value.getClass();
            }
            str = classValue.getName();
            if(value instanceof OptionHandler) {
                String options = Utils.joinOptions(((OptionHandler) value).getOptions());
                if(!options.isEmpty()) {
                    str = "\"" + str + " " + options + "\"";
                }
            }
        }
        return str;
    }

    public static void addOption(final String flag, final Collection<String> options, final Object value) {
        addFlag(flag, options);
        String str = toOptionValue(value);
        options.add(str);
    }

    public static <A> void addOption(final String flag, final Collection<String> options, final Collection<A> values) {
        for(A value : values) {
            addOption(flag, options, value);
        }
    }

    public static String flagify(String flag) {
        if(flag.charAt(0) != '-') {
            flag = "-" + flag;
        }
        return flag;
    }

    public static String unflagify(String flag) {
        if(flag.charAt(0) == '-') {
            flag = flag.substring(1);
        }
        return flag;
    }

    public static void addFlag(String flag, final Collection<String> options) {
        flag = flagify(flag);
        options.add(flag);
    }

    public static void addFlag(String flag, final Collection<String> options, boolean flagEnabled) {
        if(flagEnabled) {
            addFlag(flag, options);
        }
    }

    public static String asDirPath(String path) {
        int len = File.separator.length();
        String subStr = path.substring(path.length() - len);
        if(!subStr.equals(File.separator)) {
            return path + File.separator;
        }
        return path;
    }

    public static String asDirPath(String... path) {
        for(int i = 0; i < path.length - 1; i++) {
            path[i] = asDirPath(path[i]);
        }
        return StringUtils.join(path, "");
    }

    public static String join(final String separator, final List<String> parts) {
        if(parts.isEmpty()) {
            return "";
        }
        StringBuilder list = new StringBuilder();
        for(int i = 0; i < parts.size() - 1; i++){
            list.append(parts.get(i));
            list.append(separator);
        }
        list.append(parts.get(parts.size() - 1));
        return list.toString();
    }

    public static String indent(String string, String indentation) {
        String[] parts = string.split(System.lineSeparator());
        StringBuilder builder = new StringBuilder();
        for(String part : parts) {
            builder.append(indentation).append(part).append(System.lineSeparator());
        }
        return builder.toString();
    }

    public static String prettify(Object obj, String indentation) {
        return prettify(String.valueOf(obj), indentation);
    }

    public static String prettify(String string, String indentation) {
        int index = string.indexOf("{");
        String subStr = string;
        StringBuilder builder = new StringBuilder();
        int indentationLevel = 0;
        boolean open = true;
        while (index >= 0) {
            for(int i = 0; i < indentationLevel; i++) builder.append(indentation);
            if(open) {
                builder.append("{");
                builder.append(System.lineSeparator());
                indentationLevel++;
                for(int i = 0; i < indentationLevel; i++) builder.append(indentation);
            }
            builder.append(subStr, 0, index);
            subStr = subStr.substring(index + 1);
            if(!open) {
                builder.append(System.lineSeparator());
                for(int i = 0; i < indentationLevel; i++) builder.append(indentation);
                indentationLevel--;
                builder.append("}");
                builder.append(System.lineSeparator());
            }
            int openBraceIndex = subStr.indexOf("{");
            int closeBraceIndex = subStr.indexOf("}");
            if(openBraceIndex < 0 && closeBraceIndex < 0) {
                index = -1;
            } else open = openBraceIndex >= 0;
            if(open) {
                index = openBraceIndex;
            } else {
                index = closeBraceIndex;
            }
        }


//        String[] parts = string.split("\\{");
//        int indentationLevel = 1;
//        for(int i = 0; i < parts.length; i++) {
//            String part = parts[i];
//            for(int j = 0; j < indentationLevel; j++) {
//                builder.append(indentation);
//            }
//            builder.append("{").append(System.lineSeparator());
//            for(int j = 0; j < indentationLevel + 1; j++) {
//                builder.append(indentation);
//            }
//            part = part.replaceAll("}",System.lineSeparator() + "}" + System.lineSeparator());
//            parts[i] = part;
//            builder.append(part);
//        }
        return builder.toString();
    }

    public static boolean isFlag(String str) {
        if(str.charAt(0) == '-') {
            String subStr = str.substring(1);
            try {
                Double.parseDouble(subStr);
                return false;
            } catch (Exception e) {
                return true;
            }
        }
        return false;
    }

    public static void setOptions(OptionHandler optionHandler, final String options) throws
                                                                                             Exception {
        optionHandler.setOptions(Utils.splitOptions(options));
    }

    public static void setOptions(OptionHandler optionHandler, String options, String delimiter) throws
                                                                                                 Exception {
        String[] optionsArray = options.split(delimiter);
        optionHandler.setOptions(optionsArray);
    }
}
