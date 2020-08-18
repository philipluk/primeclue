<template>
    <div>
        <el-menu :default-active="this.$props.active" class="el-menu-demo" mode="horizontal" :router=true>
            <el-menu-item index="data-window">Data</el-menu-item>
            <el-menu-item index="classifiers-window">Classifiers</el-menu-item>
        </el-menu>
    </div>
</template>

<script>
    export default {
        props: ['active'],
    }
</script>
