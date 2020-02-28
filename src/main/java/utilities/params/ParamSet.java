package utilities.params;

import tsml.classifiers.distance_based.distances.DistanceMeasureable;
import tsml.classifiers.distance_based.distances.dtw.DTWDistance;
import utilities.StrUtils;
import weka.core.Utils;

import java.util.*;

public class ParamSet implements ParamHandler {

    private Map<String, List<Object>> paramMap = new HashMap<>();

    public List<Object> get(String name) {
        return paramMap.get(name);
    }

    public ParamSet add(String name, Object value) {
        paramMap.computeIfAbsent(name, k -> new ArrayList<>()).add(value);
        return this;
    }

    public ParamSet add(String name, ParamHandler value, List<ParamSet> params) {
        for(ParamSet paramSet : params) {
            add(name, value, paramSet);
        }
        return this;
    }

    public ParamSet add(String name, ParamHandler value, ParamSet param) {
        value.setParams(param);
        return add(name, value);
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
                list.add(StrUtils.toOptionValue(paramValue));
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
                if(value instanceof ParamHandler) {
                    ParamSet paramSet = new ParamSet();
                    paramSet.setOptions(subOptions);
                    add(flag, (ParamHandler) value, paramSet);
                } else {
                    add(flag, value);
                }
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
        return "ParamSet" +
            "{" +
//            "paramMap=" +
//                " \"" +
                StrUtils.join(", ", getOptions()) +
//                " \""
            + '}'
            ;
    }

    public static void main(String[] args) {
        ParamSet oParamSet = new ParamSet();
        oParamSet.add(DTWDistance.WARPING_WINDOW_FLAG, 3);
        ParamSet paramSet = new ParamSet();
        paramSet.add(DistanceMeasureable.DISTANCE_FUNCTION_FLAG, new DTWDistance(), oParamSet);
        String[] options;
        options = oParamSet.getOptions();
        System.out.println(Utils.joinOptions(options));
        options = paramSet.getOptions();
        System.out.println(Utils.joinOptions(options));
    }

    // todo handle lists as the value of param value --> i.e. split list into separate param values
}
