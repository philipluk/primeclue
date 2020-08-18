<template>
    <div>
        <div style="text-align: left">
            <div style="display: inline-flex; padding-top: 15px;">
                <div style="padding-top: 15px; width: 150px">
                    Field separator:
                </div>
                <div style="width: 50px">
                    <el-input v-model="fieldSeparator" @input="rewriteContent" type="text" maxlength="1"/>
                </div>
                <div style="width: 200px; margin-left: 10px; padding-top: 15px">
                    <el-checkbox v-model="firstRowIsHeader">First row as header</el-checkbox>
                </div>
            </div>
            <div>
                <el-button style="width: 350px; margin-top: 15px; margin-bottom: 15px" @click="run" size="normal" type="large">Run
                </el-button>
            </div>
        </div>
        <div style="overflow:auto; max-height:600px;">
            <table border="1" width="100%" v-html="this.table"></table>
        </div>
    </div>
</template>

<script>
    import {http} from '../../http';

    export default {
        name: "ClassificationDataTable",
        props: ['classifierName'],
        methods: {
            redrawTable(content) {
                this.content = content;
                this.outcomes = [];
                this.rewriteContent();
            },
            firstRowHasNumbers(row) {
                let columns = row.trim().split(this.fieldSeparator);
                for (let c = 0; c < columns.length; c++) {
                    if (isNaN(parseFloat(columns[c].trim()))) {
                        return false;
                    }
                }
                return true;
            },
            rewriteContent() {
                let rows = this.content.split('\n');
                this.firstRowIsHeader = !this.firstRowHasNumbers(rows[0]);
                let trs = this.getTableHeader(rows[0].trim().split(this.fieldSeparator));
                for (let i = 0; i < rows.length; i++) {
                    if (rows[i] === "") {
                        continue;
                    }
                    let tds = "";
                    let cols = rows[i].trim().split(this.fieldSeparator);
                    for (let c = 0; c < cols.length; c++) {
                        tds += `<td>${cols[c]}</td>`
                    }
                    if (this.outcomes[i] !== undefined) {
                        tds += `<td>${this.outcomes[i]}</td>`
                    }
                    if (cols.length > 0) {
                        trs += `<tr height="40">${tds}</tr>`;
                    }
                }
                this.table = trs;
            },
            getTableHeader(columns) {
                let header = `<tr height="40">`;
                for (let i = 0; i < columns.length; i++) {
                    let cb = `<input type="checkbox" id="dataColumnCheckBox${i}" title="Use column as data" checked>`;
                    header += `<th><div>${i + 1}</div><div>${cb}</div></th>`;
                }
                return header + "</tr>";
            },
            run() {
                let request = this.buildRequest();
                http.post("/classifier/classify", request)
                    .then(r => {
                        let name = request.classifier_name + ".csv";
                        http.saveAsFile(r, name);
                    })
                    .catch(e => http.generalErrorHandler(e))
            },

            buildRequest() {
                return {
                    classifier_name: this.$props.classifierName,
                    content: this.content,
                    separator: this.fieldSeparator,
                    ignore_first_row: this.firstRowIsHeader,
                    data_columns: this.getDataColumns(),
                };
            },
            getDataColumns() {
                let dataColumns = [];
                for (let i = 0; i < 1024; i++) {
                    let checkBox = document.getElementById(`dataColumnCheckBox${i}`);
                    if (checkBox === undefined || checkBox === null) {
                        return dataColumns;
                    } else {
                        dataColumns.push(checkBox.checked);
                    }
                }
            }
        },
        data() {
            return {
                firstRowIsHeader: false,
                content: "",
                fieldSeparator: ",",
                table: "",
                outcomes: [],
            }
        }
    }
</script>

<style scoped>
    table {
        border: 1px solid #ddd;
    }

    table {
        border-collapse: collapse;
        width: 100%;
    }

    td {
        padding: 15px;
        text-align: left;
    }

    th {
        border: 1px solid #ddd;
        padding: 15px;
        text-align: center;
    }

</style>