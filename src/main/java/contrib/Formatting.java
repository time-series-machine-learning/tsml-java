package contrib;

import experiments.data.DatasetLoading;
import fileIO.InFile;
import fileIO.OutFile;
import weka.core.Instance;
import weka.core.Instances;

import java.io.IOException;

public class Formatting {


    public static void main(String[] args) {
        converterARFFtoTS();
    }

    public static void formatTiselac(){
        InFile inf =new InFile("C:\\Temp\\Tiselac\\Tiselac_TRAIN.csv");
        InFile inf2 =new InFile("C:\\Temp\\Tiselac\\Tiselac_TEST.csv");
        OutFile out =new OutFile("C:\\Temp\\Tiselac\\Tiselac_TRAIN.arff");
        OutFile out2 =new OutFile("C:\\Temp\\Tiselac\\Tiselac_TEST.arff");
        String str = inf.readLine();
/*        while(str!=null){
            String[] split = str.split(",");
            for(int i=0;i< split.length-2;i++)
                out.writeString(split[i]+",");
            String lastData=split[split.length-2].replace("\\n","");
            System.out.println(lastData);
            out.writeString(lastData+",");
            out.writeLine(split[split.length-1]);
            str=inf.readLine();
        }
*/
        str = inf2.readLine();
        while(str!=null) {
            String[] split = str.split(",");
            for (int i = 0; i < split.length - 2; i++)
                out2.writeString(split[i] + ",");
            String lastData = split[split.length - 2].replace("\\n", "");
            System.out.println(lastData);
            out2.writeString(lastData + ",");
            out2.writeLine(split[split.length - 1]);
            str = inf2.readLine();
        }

    }
    static String[] newProblem={"AbnormalHeartbeat","AconityMINIPrinterLargeEq", "AconityMINIPrinterSmallEq","AsphaltObstaclesUniEq","AsphaltPavementTypeUniEq",
    "AsphaltRegularityUniEq","BinaryHeartbeat","CatsDogs","Colposcopy","DucksaAndGeese","ElectricDeviceDetection","FruitFlies","KeplerLightCurves","MITBIH-Heartbeat",
            "RightWhaleCalls","SharePriceIncrease","UrbanSound"};
    static String path = "C:\\Data\\NewUnivariateDatasets\\";
    public static void converterARFFtoTS(){
        String head = "";
        for(int i=0;i< newProblem.length;i++){
            try {
                System.out.println(" Converting  = "+newProblem[i]);
                Instances train = DatasetLoading.loadData(path+newProblem[i]+"\\"+newProblem[i]+"_TRAIN");
                Instances test = DatasetLoading.loadData(path+newProblem[i]+"\\"+newProblem[i]+"_TEST");
                OutFile tsTrain = new OutFile(path+newProblem[i]+"\\"+newProblem[i]+"_TRAIN.ts");
                OutFile tsTest = new OutFile(path+newProblem[i]+"\\"+newProblem[i]+"_TEST.ts");
                head+="@problemName " + train.relationName();
                head+="\n@timeStamps false";
                head+="\n@missing  false";
                head+="\n@univariate true";
                head+="\n@equalLength true";
                head+="\n@seriesLength " + (test.numAttributes()-1);
                //outW.println("@classLabel " + );
                head+="\n@classLabel true ";
                for(int j=0;j<train.numClasses();j++)
                    head+=j+" ";
                head+="\n@data";
                tsTrain.writeLine(head);
                tsTest.writeLine(head);
                for(Instance ins:train){
                    double[] atts=ins.toDoubleArray();
                    for(int j=0;j<atts.length-2;j++)
                        tsTrain.writeString(atts[j]+",");
                    tsTrain.writeString(atts[atts.length-2]+":");
                    tsTrain.writeLine((int)(atts[atts.length-1])+"");
                }
                System.out.println(" Finished train");
                for(Instance ins:test){
                    double[] atts=ins.toDoubleArray();
                    for(int j=0;j<atts.length-2;j++)
                        tsTest.writeString(atts[j]+",");
                    tsTest.writeString(atts[atts.length-2]+":");
                    tsTest.writeLine((int)(atts[atts.length-1])+"");
                }
                System.out.println(" Finished test");

            } catch (IOException e) {
                System.out.println(" Failed to load "+newProblem[i]);
            }
        }
    }



}
