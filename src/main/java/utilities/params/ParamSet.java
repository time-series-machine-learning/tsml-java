package utilities.params;

import java.util.*;

public class ParamSet {
    public static class ParamValue {
        private Object value;
        private List<ParamSet> paramSetList = new ArrayList<>();

        public ParamValue() {}

        public ParamValue(Object value, List<ParamSet> paramSetList) {
            setParamSetList(paramSetList);
            setValue(value);
        }

        public ParamValue(Object value, ParamSet paramSet) {
            this(value, new ArrayList<>(Collections.singletonList(paramSet)));
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

        public List<ParamSet> getParamSetList() {
            return paramSetList;
        }

        public void setParamSetList(List<ParamSet> paramSetList) {
            if(paramSetList == null) {
                paramSetList = new ArrayList<>();
            }
            this.paramSetList = paramSetList;
        }

        public void addParam(final ParamSet paramSet) {
            paramSetList.add(paramSet);
        }

        @Override public String toString() {
            return "ParamValue{" +
                "value=" + value +
                ", paramList=" + paramSetList +
                '}';
        }
    }

    private Map<String, List<ParamValue>> paramMap = new HashMap<>();

    public List<ParamValue> get(String name) {
        return paramMap.get(name);
    }

    public ParamSet add(String name, Object value) {
        return add(name, new ParamValue(value));
    }

    public ParamSet add(String name, Object value, List<ParamSet> paramSets) {
        return add(name, new ParamValue(value, paramSets));
    }

    public ParamSet add(String name, Object value, ParamSet paramSet) {
        return add(name, new ParamValue(value, paramSet));
    }

    public ParamSet add(String name, ParamValue value) {
        paramMap.computeIfAbsent(name, k -> new ArrayList<>()).add(value);
        return this;
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
//        Param param = new Param();
//        Param wParam = new Param();
//        wParam.add(Dtw.WARPING_WINDOW_FLAG, new ParamValue(4));
//        param.add(DistanceMeasure.DISTANCE_FUNCTION_FLAG, new ParamValue(new Dtw(), wParam));

    }
}
