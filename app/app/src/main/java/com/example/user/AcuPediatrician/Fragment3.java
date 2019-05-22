package com.example.user.AcuPediatrician;
import android.content.Intent;
import android.os.Bundle;
import android.support.annotation.Nullable;
import android.support.v4.app.Fragment;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.TextView;

public class Fragment3 extends Fragment  {
    private TextView textView;
    private Button button;

    @Nullable
    @Override
    public View onCreateView(LayoutInflater inflater, @Nullable ViewGroup container, @Nullable Bundle savedInstanceState) {
        View view=inflater.inflate(R.layout.fragment3,container,false);
        return view;
    }
    @Override
    public void onActivityCreated(@Nullable Bundle savedInstanceState) {
        super.onActivityCreated(savedInstanceState);
        Button button6_1 =(Button) getView().findViewById(R.id.button6_1);
        button6_1.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(getActivity(),Toulianbu.class);
                startActivity(intent);
            }
        });
        Button button6_2 =(Button) getView().findViewById(R.id.button6_2);
        button6_2.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(getActivity(),Yaofubu.class);
                startActivity(intent);
            }
        });
        Button button6_3 =(Button) getView().findViewById(R.id.button6_3);
        button6_3.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(getActivity(),Beibu.class);
                startActivity(intent);
            }
        });
        Button button6_4 =(Button) getView().findViewById(R.id.button6_4);
        button6_4.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(getActivity(),Sizi.class);
                startActivity(intent);
            }
        });

    }
}