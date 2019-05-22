package com.example.user.AcuPediatrician;

import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.webkit.WebView;
import android.widget.Button;
import android.widget.TextView;

public class Jingfeng extends AppCompatActivity {
    private TextView textView;
    private Button button;
    private WebView webview;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.jingfeng);
        WebView webview = (WebView)findViewById(R.id.webview);
        webview.getSettings().setJavaScriptEnabled(true);
        webview.loadUrl("https://www.zk120.com/zixun/p/51392.html?shprefix=xetn-gd");
    }
}
