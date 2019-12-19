package utilities.params;

import scala.annotation.meta.param;
import tsml.classifiers.distance_based.distances.DistanceMeasure;
import tsml.classifiers.distance_based.distances.Dtw;
import tsml.classifiers.distance_based.distances.Lcss;
import utilities.StrUtils;
import weka.core.OptionHandler;
import weka.core.Utils;

import java.util.*;
import java.util.function.BiConsumer;

public class ParamSet implements ParamHandler {
    public static class ParamValue {
        private Object value;
        private List<ParamSet> paramList = new ArrayList<>();

        public ParamValue() {}

        public ParamValue(Object value, List<ParamSet> paramList) {
            setParamList(paramList);
            setValue(value);
        }

        public ParamValue(Object value, ParamSet param) {
            this(value, new ArrayList<>(Arrays.asList(param)));
        }

        public ParamValue(Object value) {
            this(value, new ArrayList<>()); // no sub param
        }

        public Object getValue() {
            return value;
        }

        public void setValue(final Object value) {
            this.value = value;
        }

        public List<ParamSet> getParamList() {
            return paramList;
        }

        public void setParamList(List<ParamSet> paramList) {
            if(paramList == null) {
                paramList = new ArrayList<>();
            }
            this.paramSetList = paramSetList;
        }

        public void addParam(final ParamSet param) {
            paramList.add(param);
        }

        @Override public String toString() {
            return "ParamValue{" +
                "value=" + value +
                ", paramList=" + paramSetList +
                '}';
        }

        private List<String> getOptionsList() {
            List<String> list = new ArrayList<>();
            list.add(StrUtils.toOptionValue(value));
            for(ParamSet paramSet : paramList) {
                list.addAll(paramSet.getOptionsList());
            }
            return list;
        }
    }

    private Map<String, List<ParamValue>> paramMap = new HashMap<>();

    public List<ParamValue> get(String name) {
        return paramMap.get(name);
    }

    public ParamSet add(String name, Object value) {
        return add(name, new ParamValue(value));
    }

    public ParamSet add(String name, Object value, List<ParamSet> params) {
        return add(name, new ParamValue(value, params));
    }

    public ParamSet add(String name, Object value, ParamSet param) {
        return add(name, new ParamValue(value, param));
    }

    public ParamSet add(String name, ParamValue value) {
        paramMap.computeIfAbsent(name, k -> new ArrayList<>()).add(value);
        return this;
    }

    public ParamSet addAll(ParamSet paramSet) {
        paramSet.paramMap.forEach(this::add);
        return this;
    }

    public ParamSet clear() {
        paramMap.clear();
        return this;
    }

    @Override public List<String> getOptionsList() {
        List<String> list = new ArrayList<>();
        paramMap.forEach((name, paramValues) -> {
            paramValues.forEach(paramValue -> {
                list.add(StrUtils.flagify(name));
                List<String> optionsList = paramValue.getOptionsList();
                String options = StrUtils.joinOptions(optionsList);
                list.add(options);
            });
        });
        return list;
    }

    // todo make paramSet / paramSpace compatible with flags (urgh flags are naff)

    @Override public void setOptionsList(final List<String> options) throws
                                                                     Exception {
        for(int i = 0; i < options.size(); i++) {
            String option = options.get(i);
            String flag = StrUtils.unflagify(option);
            if(StrUtils.isOption(option, options)) {
                String[] subOptions = Utils.splitOptions(option);
                String optionValue = subOptions[0];
                subOptions[0] = "";
                Object value = StrUtils.fromOptionValue(optionValue);
                ParamSet paramSet = new ParamSet();
                paramSet.setOptions(subOptions);
                add(flag, new ParamValue(value, paramSet));
                options.set(i, "");
                i++;
            } else {
                add(flag, true);
            }
            options.set(i, "");
        }
    }

    @Override public ParamSet getParams() {
        return this;
    }

    @Override public void setParams(final ParamSet param) {
        param.paramMap.forEach(this::add);
    }

    @Override public String toString() {
        return "Param" +
//            "{" +
//            "paramMap=" +
            paramMap
//            +
//            '}'
            ;
    }

    public static void main(String[] args) {
        ParamSet oParamSet = new ParamSet();
        oParamSet.add(Dtw.WARPING_WINDOW_FLAG, 3);
        ParamSet paramSet = new ParamSet();
        paramSet.add(DistanceMeasure.DISTANCE_FUNCTION_FLAG, new Dtw(), oParamSet);
        String[] options;
        options = oParamSet.getOptions();
        System.out.println(Utils.joinOptions(options));
        options = paramSet.getOptions();
        System.out.println(Utils.joinOptions(options));
    }

    // todo handle lists as the value of param value --> i.e. split list into separate param values
}
