package com.example.user.AcuPediatrician;

import android.content.Intent;
import android.os.Bundle;
import android.support.annotation.Nullable;
import android.support.v7.app.AppCompatActivity;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;

public class Yuerbaojian extends AppCompatActivity {
    private TextView textView;
    private Button button;

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.yuerbaojian);
        Button button5 = (Button) findViewById(R.id.button5);
        button5.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent1 = new Intent(Yuerbaojian.this, Yanshi.class);
                startActivity(intent1);
            }
        });
        Button button5_1 = (Button) findViewById(R.id.button5_1);
        button5_1.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent2 = new Intent(Yuerbaojian.this, Shizhen.class);
                startActivity(intent2);
            }
        });
        Button button5_2 = (Button) findViewById(R.id.button5_2);
        button5_2.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent3 = new Intent(Yuerbaojian.this, Pifuhuli.class);
                startActivity(intent3);
            }
        });
    }
}
