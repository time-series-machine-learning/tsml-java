package evaluation;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;


class P{
    {
        System.out.println("A");
    }
    static {
        System.out.println("B");
    }
}

public class TestTS extends P{

    static String file = "C:/Users/fbu19zru/code/test_day.csv";
    static String fileClass = "C:/Users/fbu19zru/code/class_test_day.csv";
    static String arffFile = "C:/Users/fbu19zru/code/test_day.ts";



    public static void main(String[] arg){


        try{
            FileWriter outFile = new FileWriter(arffFile);
            writeHeader(outFile);
            writeBody(outFile);
            outFile.close();
        }catch (Exception e){
            e.printStackTrace();
        }




    }
    private static void writeHeader(FileWriter fw) throws IOException {


        fw.write("@problemName MEGMind\n");
        fw.write("@timeStamps false\n");
        fw.write("@missing false\n");
        fw.write("@univariate false\n");
        fw.write("@dimensions 204\n");
        fw.write("@equalLength true\n");
        fw.write("@seriesLength 200\n");
        fw.write("@classLabel true 1 2 3 4 5\n\n");
        fw.write("@data\n");


    }
    private static void writeBody(FileWriter fw){
        DecimalFormat df = new DecimalFormat("#");
        df.setMaximumFractionDigits(20);
        try (BufferedReader br = new BufferedReader(new FileReader(file))) {
            BufferedReader brClass = new BufferedReader(new FileReader(fileClass));
            String line;
            String lineClass;
            while ((line = br.readLine()) != null) {
                lineClass = brClass.readLine();
                String[] spl = line.split(",");


                for (int i=0;i<204;i++){
                    String[] subset = Arrays.copyOfRange(spl, i*200, (i+1)*200);
                    fw.write(Arrays.toString(subset).replace("[", "")  //remove the right bracket
                            .replace("]", "").trim() );
                    fw.write(":");
                }
                fw.write(lineClass+"\n");


            }


        }catch (Exception e){
            e.printStackTrace();
        }
    }
}
