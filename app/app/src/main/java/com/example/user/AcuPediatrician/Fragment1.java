package com.example.user.AcuPediatrician;


import android.content.Intent;
import android.os.Bundle;
import android.support.annotation.Nullable;
import android.support.v4.app.Fragment;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.webkit.WebView;
import android.widget.Button;
import android.widget.TextView;
import android.os.Handler;
import android.support.v4.view.PagerAdapter;
import android.support.v4.view.ViewPager;
import android.widget.ImageView;
import android.widget.LinearLayout;

import java.util.ArrayList;


public class Fragment1 extends Fragment {
    private TextView textView;
    private Button button;
    private ViewPager viewPager;
    private WebView webview;
    private final int[] imageIds = { R.drawable.tuina_1, R.drawable.tuina_2, R.drawable.tuina_3, R.drawable.tuina_4 };

    private ArrayList<ImageView> imageList;

    protected int lastPosition;
    @Nullable
    @Override
    public View onCreateView(LayoutInflater inflater, @Nullable ViewGroup container, @Nullable Bundle savedInstanceState) {
        View view=inflater.inflate(R.layout.fragment1,container,false);
        return view;
    }
    @Override
    public void onActivityCreated(@Nullable Bundle savedInstanceState) {
        super.onActivityCreated(savedInstanceState);
        Button button3 =(Button) getView().findViewById(R.id.button3);
        button3.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(getActivity(),Yuerbaojian.class);
                startActivity(intent);
            }
        });
        Button button =(Button) getView().findViewById(R.id.button);
        button.setOnClickListener(new View.OnClickListener() {
                @Override
                public void onClick(View v) {
                    Intent intent = new Intent(getActivity(),Tizhiceshi.class);
                    startActivity(intent);
                }
        });
        Button button2 =(Button) getView().findViewById(R.id.button2);
        button2.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(getActivity(),LoginActivity.class);
                startActivity(intent);
            }
        });
        Button button2_1 =(Button) getView().findViewById(R.id.button2_1);
        button2_1.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(getActivity(),Yeti.class);
                startActivity(intent);
            }
        });
        Button button2_2 =(Button) getView().findViewById(R.id.button2_2);
        button2_2.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(getActivity(),Xinshengerhuangdan.class);
                startActivity(intent);
            }
        });
        Button button2_3 =(Button) getView().findViewById(R.id.button2_3);
        button2_3.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(getActivity(),Jingfeng.class);
                startActivity(intent);
            }
        });
        Button button2_4 =(Button) getView().findViewById(R.id.button2_4);
        button2_4.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(getActivity(),Yiniao.class);
                startActivity(intent);
            }
        });
        Button button2_5 =(Button) getView().findViewById(R.id.button2_5);
        button2_5.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(getActivity(),Ekouchuang.class);
                startActivity(intent);
            }
        });
        Button button2_6 =(Button) getView().findViewById(R.id.button2_6);
        button2_6.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(getActivity(),Outu.class);
                startActivity(intent);
            }
        });
        Button button2_7 =(Button) getView().findViewById(R.id.button2_7);
        button2_7.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(getActivity(),Shiji.class);
                startActivity(intent);
            }
        });
        Button button2_8 =(Button) getView().findViewById(R.id.button2_8);
        button2_8.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(getActivity(),Fuxie.class);
                startActivity(intent);
            }
        });
        Button button2_9 =(Button) getView().findViewById(R.id.button2_9);
        button2_9.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(getActivity(),Yanshi1.class);
                startActivity(intent);
            }
        });


                viewPager = (ViewPager) getView().findViewById(R.id.viewpager);

        imageList = new ArrayList<>();
        for (int i = 0; i < imageIds.length; i++) {
            ImageView image = new ImageView(getActivity());
            image.setBackgroundResource(imageIds[i]);
            imageList.add(image);
            ImageView point = new ImageView(getActivity());
            LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.WRAP_CONTENT,
                    LinearLayout.LayoutParams.WRAP_CONTENT);

            params.rightMargin = 20;
            point.setLayoutParams(params);

            point.setBackgroundResource(R.drawable.ui);
            if (i == 0) {
                point.setEnabled(true);
            } else {
                point.setEnabled(false);
            }

        }
        viewPager.setAdapter(new Fragment1.MyPagerAdapter());
        viewPager.setOnPageChangeListener(new ViewPager.OnPageChangeListener() {

            @Override
            public void onPageSelected(int position) {

                position = position % imageList.size();
                lastPosition = position;
            }
            @Override
            public void onPageScrolled(int position, float positionOffset,
                                       int positionOffsetPixels) {
            }
            @Override
            public void onPageScrollStateChanged(int state) {
            }
        });
        isRunning = true;
        handler.sendEmptyMessageDelayed(0, 3000);
    }

    private boolean isRunning = false;

    private Handler handler = new Handler() {
        public void handleMessage(android.os.Message msg) {
            viewPager.setCurrentItem(viewPager.getCurrentItem() + 1);
            if (isRunning) {
                handler.sendEmptyMessageDelayed(0, 3000);
            }
        };
    };
    @Override
    public void onDestroyView() {
        super.onDestroyView();
        isRunning = false;
    }

    private class MyPagerAdapter extends PagerAdapter {
        @Override
        public int getCount() {
            return Integer.MAX_VALUE; // 使得图片可以循环
        }
        @Override
        public Object instantiateItem(ViewGroup container, int position) {
            container.addView(imageList.get(position % imageList.size()));
            return imageList.get(position % imageList.size());
        }
        @Override
        public boolean isViewFromObject(View view, Object object) {
            if (view == object) {
                return true;
            } else {
                return false;
            }
        }
        @Override
        public void destroyItem(ViewGroup container, int position, Object object) {
            container.removeView((View) object);
        }
    }

    }
