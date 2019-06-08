import timeseriesweka.filters.DerivativeFilter;
import utilities.ClassifierTools;
import weka.core.Instances;
import weka.filters.Filter;

public class Main {

    public static void main(String[] args) throws Exception {
        Filter filter = new DerivativeFilter();
        Instances train = ClassifierTools.loadData("/home/goastler/Univariate2018/GunPoint/GunPoint_TRAIN.arff");
        filter.setInputFormat(train);
        Instances derTrain = Filter.useFilter(train, filter);
        boolean a = true;
    }
}
