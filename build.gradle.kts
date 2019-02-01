plugins {
    java
    application
}

repositories {
    mavenCentral()    
    jcenter() 
}

dependencies {
    testImplementation("junit:junit:4.12")
    implementation(fileTree("lib"))
}

application {
    mainClassName = "App"
}
