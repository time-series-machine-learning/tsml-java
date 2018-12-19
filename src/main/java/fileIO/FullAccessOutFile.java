package fileIO;
import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.attribute.PosixFilePermission;
import java.util.TreeSet;

public class FullAccessOutFile extends OutFile{

    public FullAccessOutFile(String n) {
        super(n);
    }
    @Override
    public void closeFile() {
        outFile.close();
        File f=new File(name);
        TreeSet<PosixFilePermission> perms=new TreeSet<>();
        for(PosixFilePermission p:PosixFilePermission.values())
        perms.add(p);
        Path path=f.toPath();
        try{
            Files.setPosixFilePermissions(path, perms);
        }catch(Exception e){
            System.out.println("UNABLE TO CHANGE PERMISSIONS FOR FILE "+name);
        }

    }
}
	