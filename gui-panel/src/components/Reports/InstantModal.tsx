import React, {useEffect, useState} from "react";
import {KPI} from "../../api/DataStructures";
import PersistentDataManager from "../../api/DataManager";
import FilterOptions from "../Selectors/FilterOptions";

interface ModalProps {
    isOpen: boolean;
    reportData: Partial<{
        name: string;
        email: string;
        period: string;
        kpis: string[];
        machines: string[];
        machineType?: string
    }>;
    onSave: (reportData: Partial<{ name: string; email: string; period: string; kpis: string[]; machines: string[]; // List of selected machines
        machineType?: string }>) => void;
    onClose: () => void;
}

const ReportModal: React.FC<ModalProps> = ({isOpen, reportData, onSave, onClose}) => {
    const dataManager = PersistentDataManager.getInstance();
    const [formData, setFormData] = useState(reportData);
    const [validationErrors, setValidationErrors] = useState<{ [key: string]: string }>({});

    useEffect(() => {
        setFormData(reportData); // Initialize the form with the passed report data
    }, [reportData]);

    const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
        const {name, value} = e.target;
        setFormData((prev) => ({...prev, [name]: value}));

        // Clear validation error on valid input
        setValidationErrors((prev) => ({...prev, [name]: ""}));
    };

    const [isFormValid, setIsFormValid] = useState(false);

    useEffect(() => {
        // Validate the form whenever `formData` changes
        const errors: { [key: string]: string } = {};

        if (!formData.name?.trim()) errors.name = "Report name is required.";
        if (!formData.email?.trim() || !/\S+@\S+\.\S+/.test(formData.email))
            errors.email = "A valid email is required.";
        if (!formData.period) errors.period = "Report period is required.";
        if (!formData.kpis || formData.kpis.length === 0) errors.kpis = "At least one KPI is required.";
        if (!formData.machines || formData.machines.length === 0)
            errors.machines = "At least one machine is required.";

        setValidationErrors(errors);
        setIsFormValid(Object.keys(errors).length === 0);
    }, [formData]);

    const handleSave = () => {
        if (isFormValid) {
            onSave(formData);
        }
    };

    if (!isOpen) return null;

    return (
        <div
            className="fixed inset-0 bg-gray-500 bg-opacity-50 flex items-center justify-center overflow-y-auto"
            style={{zIndex: 1000}}
        >
            <div
                className="bg-white p-6 rounded-lg shadow-lg max-w-3xl w-full space-y-4 px-4"
                style={{
                    maxHeight: "90vh", // Restrict modal height to 90% of the viewport
                    overflowY: "auto", // Add scroll for overflowing content
                }}
            >
                <h2 className="text-xl font-semibold mb-4">Request Report</h2>

                {/* Report Name */}
                <label className="block mb-2 text-sm">Report Name</label>
                <input
                    type="text"
                    name="name"
                    value={formData.name || ""}
                    onChange={handleInputChange}
                    className={`w-full p-2 mb-1 border rounded ${
                        validationErrors.name ? "border-red-500" : "border-gray-300"
                    }`}
                    placeholder="Enter report name"
                />
                {validationErrors.name && <p className="text-red-500 text-sm">{validationErrors.name}</p>}

                {/* Report Period */}
                <label className="block mb-2 text-sm">Period</label>
                <select
                    name="period"
                    value={formData.period || "Daily"}
                    onChange={handleInputChange}
                    className="w-full p-2 mb-4 border border-gray-300 rounded"
                >
                    <option value="Week">Weekly</option>
                    <option value="Month">Monthly</option>
                    <option value="Year">Yearly</option>
                </select>
                {validationErrors.period && <p className="text-red-500 text-sm">{validationErrors.period}</p>}

                {/* KPI Selection */}
                <label className="block mb-2 text-sm font-bold text-gray-700">KPIs to Include</label>
                <div className="border rounded p-3 mb-1">
                    {Object.entries(
                        dataManager.getKpiList().reduce((groups, kpi) => {
                            if (!groups[kpi.type]) {
                                groups[kpi.type] = [];
                            }
                            groups[kpi.type].push(kpi);
                            return groups;
                        }, {} as Record<string, KPI[]>)
                    ).map(([type, kpis]) => (
                        <div key={type} className="mb-4">
                            <h3 className="text-sm font-bold text-gray-800 mb-2">{type}</h3>
                            <div className="grid grid-cols-2 md:grid-cols-3 gap-x-4 gap-y-2 font-semibold">
                                {kpis.map((kpi) => (
                                    <div key={kpi.id} className="flex items-center text-start mb-2">
                                        <input
                                            type="checkbox"
                                            id={`kpi-${kpi.id}`}
                                            value={kpi.id}
                                            checked={formData.kpis?.includes(kpi.id) || false}
                                            onChange={(e) => {
                                                const selectedKpis = formData.kpis || [];
                                                if (e.target.checked) {
                                                    setFormData((prev) => ({
                                                        ...prev,
                                                        kpis: [...selectedKpis, kpi.id],
                                                    }));
                                                } else {
                                                    setFormData((prev) => ({
                                                        ...prev,
                                                        kpis: selectedKpis.filter((id) => id !== kpi.id),
                                                    }));
                                                }
                                            }}
                                            className="mr-2 h-4 w-4 border-gray-300 rounded text-blue-600 focus:ring-blue-500"
                                        />
                                        <label htmlFor={`kpi-${kpi.id}`} className="text-sm text-gray-700">
                                            {kpi.name}
                                        </label>
                                    </div>
                                ))}
                            </div>
                        </div>
                    ))}
                </div>
                {validationErrors.kpis && <p className="text-red-500 text-sm">{validationErrors.kpis}</p>}
                {/* Machines */}
                <label className="block mb-2 text-sm">Machines</label>
                <FilterOptions
                    filter={{
                        machineType: formData.machineType || "Custom Machine Set",
                        machineIds: formData.machines || [],
                    }}
                    onChange={(updatedFilter) =>
                        setFormData((prev) => ({
                            ...prev,
                            machineType: updatedFilter.machineType,
                            machines: updatedFilter.machineIds,
                        }))
                    }
                />
                {validationErrors.machines && <p className="text-red-500 text-sm">{validationErrors.machines}</p>}

                {/* Email */}
                <label className="block mb-2 text-sm">Email</label>
                <input
                    type="email"
                    name="email"
                    value={formData.email || ""}
                    onChange={handleInputChange}
                    className={`w-full p-2 mb-1 border rounded ${
                        validationErrors.email ? "border-red-500" : "border-gray-300"
                    }`}
                    placeholder="Enter email"
                />
                {validationErrors.email && <p className="text-red-500 text-sm">{validationErrors.email}</p>}

                {/* Actions */}
                <div className="flex space-x-4 mt-4">
                    <button
                        className="px-4 py-2 bg-gray-500 text-white rounded-lg hover:bg-gray-600"
                        onClick={onClose}
                    >
                        Cancel
                    </button>
                    <button
                        className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600"
                        onClick={handleSave}
                        disabled={!isFormValid}
                    >
                        Request Report
                    </button>
                </div>
            </div>
        </div>
    );
};

export default ReportModal;
